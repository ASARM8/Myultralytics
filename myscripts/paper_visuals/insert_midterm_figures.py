"""Insert supplied method slides and archived experiment figures into the report.

Preserves existing paragraphs; adds captions, native tables and result explanations.
Does not render or visually audit the supplied slides or the Word document.
"""
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import csv
import hashlib
import json
import shutil

from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'mydocs/中期报告'
ASSETS = BASE / '图表补充_20261114'
REPORT = BASE / '中期-初稿.docx'
WIRE = ROOT / 'res/innovation2/wire_seg'


def main():
    doc = Document(REPORT)
    originals = list(doc.element.body.iter(qn('w:p')))
    old_text = [Paragraph(p, doc).text for p in originals]
    if any('图1 检测精修引导的电线实例分割整体流程' == t for t in old_text):
        raise SystemExit('This report already contains this insertion batch; refusing duplicates.')
    backup = BASE / ('中期-初稿_插图前备份_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.docx')
    shutil.copy2(REPORT, backup)
    changes = []
    sources = {}
    def record_source(p):
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    def anchor(prefix):
        matches = [p for p in originals if Paragraph(p, doc).text.startswith(prefix)]
        assert len(matches) == 1, (prefix, len(matches))
        return matches[0]
    body_format = deepcopy(anchor('在无人机低空图像中，电线通常具有较大的长宽比。').find(qn('w:pPr')))
    def para(after, text='', centered=False, picture=False):
        node = OxmlElement('w:p')
        after.addnext(node)
        p = Paragraph(node, doc)
        if not centered and not picture and body_format is not None:
            node.append(deepcopy(body_format))
        if centered or picture:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(6)
        if picture:
            p.paragraph_format.keep_with_next = True
        if text:
            r = p.add_run(text)
            if centered:
                r.font.size = Pt(10.5)
        return node
    def figure(after, path, caption, description):
        path = record_source(path)
        node = para(after, picture=True)
        Paragraph(node, doc).add_run().add_picture(str(path), width=Cm(15.6))
        node = para(node, caption, centered=True)
        node = para(node, description)
        changes.append({'type':'figure', 'caption':caption, 'source':str(path.relative_to(ROOT))})
        return node
    def table(after, caption, headers, rows, note):
        node = para(after, caption, centered=True)
        Paragraph(node, doc).paragraph_format.keep_with_next = True
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = 'Table Grid'
        for c, value in zip(t.rows[0].cells, headers):
            c.text = value
        for row in rows:
            for c, value in zip(t.add_row().cells, row):
                c.text = str(value)
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.first_line_indent = Cm(0)
                    p.paragraph_format.line_spacing = 1.0
                    for run in p.runs:
                        run.font.size = Pt(9)
        for c in t.rows[0].cells:
            for r in c.paragraphs[0].runs:
                r.bold = True
        repeat = OxmlElement('w:tblHeader')
        t.rows[0]._tr.get_or_add_trPr().append(repeat)
        node.addnext(t._tbl)
        changes.append({'type':'table', 'caption':caption,'rows':rows})
        return para(t._tbl, note)
    def csvrows(name):
        p=record_source(ASSETS/'tables'/name)
        with p.open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))

    node=anchor('检测器、精修器和分割器分别训练后，可分别检验')
    node=para(node,'整体处理流程如图1所示。检测与尺度精修提供候选实例的几何信息，分割阶段据此从原始图像中提取定向条带，并结合目标与邻线提示预测可见掩码。三个模块采用分阶段训练，推理时通过候选编号及坐标变换保持实例对应关系。')
    figure(node,ASSETS/'方法图导出/method_1.png','图1 检测精修引导的电线实例分割整体流程','图中展示从输入图像、旋转框检测和尺度精修，到定向裁剪、提示构造及掩码输出的处理关系。输出为二维旋转框及可见实例掩码，为低空障碍物感知提供目标级和像素级信息。')

    node=anchor('整个流程先通过回归范围感知分配筛选几何上可回归的候选点')
    figure(node,ASSETS/'方法图导出/method_2.png','图2 回归范围感知分配与候选尺度精修','图2将训练时的范围约束和推理时的尺度更新分别表示。真实框用于训练分配及残差监督；在线精修依据预测候选与P2、P3局部特征完成，保持候选中心、角度和置信度不变。')
    node=anchor('在当前验证集上，采用回归范围感知配置后')
    det=csvrows('02_detection_results.csv')
    node=table(node,'表1 旋转框检测阶段的对照结果',['方法','Precision/%','Recall/%','mAP50/%','mAP50–95/%'],[[r['方法'],*[f'{float(r[k]):.3f}' for k in ('Precision (%)','Recall (%)','OBB mAP50 (%)','OBB mAP50–95 (%)')]] for r in det],'表1中CA为扩展回归区间与回归范围感知分配的联合配置，精修残差倍率为0.42。CA相对基线的OBB mAP50–95提高4.574个百分点，加入精修后再提高4.167个百分点；此表反映阶段组合效果，不能单独分解两项训练调整的贡献。')
    figure(node,ASSETS/'figures/09a_detection_examples.png','图3 旋转框尺度精修的改善与退化案例','绿色为真实旋转框，橙色为预测框；各行使用相同局部图像窗口。前两行对应定位改善，第三行保留精修后的退化情形。图中ΔIoU沿用完整图块的匹配统计，表明尺度更新的总体收益仍伴随个别候选退化。')

    node=anchor('当指定的分割对象改变时，同一候选的轴线可以')
    figure(node,ASSETS/'方法图导出/method_3.png','图4 目标与邻线角色化有限轴线提示分割流程','图4展示单个候选的定向裁剪、目标提示、邻线提示、五通道输入和掩码恢复。目标及邻线分别使用独立提示通道；真实实例标注仅用于监督或结果对照，在线提示由预测旋转框自动生成。')

    metrics=[]
    for kind in ('target','points','boxes','axes'):
        p=record_source(WIRE/kind/'metrics_val.json')
        m=json.loads(p.read_text(encoding='utf-8'))
        assert m['prompt']==kind and m['swap_target_pairs']==342
        s=m['matched_with_neighbors']
        metrics.append([kind,f"{100*m['mask_mAP50_95']:.3f}",f"{100*s['target_pixel_recall']:.3f}",f"{100*s['neighbor_pixel_leakage']:.3f}",f"{round(m['swap_target_pairs']*m['swap_both_correct_fraction'])}/342",f"{100*m['swap_both_correct_fraction']:.3f}"])
    node=anchor('角色互换试验根据相对 IoU 判断模型选中了哪个实例')
    node=table(node,'表2 四种邻线提示的分割质量与角色互换结果',['提示','Mask mAP/%','目标召回/%','邻线泄漏/%','互换成功次数','互换成功率/%'],metrics,'掩码二值化阈值为0.5。Mask mAP采用完整验证集的IoU 0.50–0.95均值；召回和泄漏统计386个邻线暴露匹配实例；互换统计342次有方向的目标—邻线试验。互换成功要求两次输出分别更接近各自目标，不额外要求绝对掩码IoU超过0.5。')
    para(node,'表2中，axes的互换成功率为66.959%，比target、points和boxes分别高30.409、5.556和9.064个百分点。其整体Mask mAP为46.961%，与points、boxes接近。上述结果分别刻画指定实例选择和常规掩码质量，应结合评价条件解释。')

    node=anchor('在目标像素召回率相近的实测工作点')
    node=figure(node,ASSETS/'figures/07a_recall_leakage_threshold.png','图5 目标召回、邻线泄漏与掩码阈值的关系','左图统计386个邻线暴露实例，点对应0.1至0.5的实际掩码阈值；右图统计完整1695张验证图块的Mask mAP50–95。四组方法的最高采样mAP均出现在阈值0.4，axes与points分别为47.601%和47.526%。曲线连接用于展示工作点变化，不代表额外测得了中间阈值。')
    node=figure(node,ASSETS/'figures/07b_neighbor_pair_threshold.png','图6 不同掩码阈值下的双实例分割成功邻线对数','以270对真实邻线为共同分母，两根电线的预测掩码均达到指定IoU时才计为成功。阈值0.5、IoU门限0.5时，target为130对，其余三组均为129对；掩码阈值降至0.2时，axes为158对，target为160对。该统计与表2的相对身份互换指标采用不同判定条件。')
    figure(node,ASSETS/'figures/09b_segmentation_failure_examples.png','图7 分割阈值变化后的恢复与失败案例','同一axes概率图分别按0.5和0.2二值化，各行展示同一256×256局部区域，标注IoU按完整640×640图块中的实例计算。三行分别为降低阈值后恢复、仍明显漏分和误分增加的案例。在37个失败实例中，90.22%的遗漏目标像素位于裁剪内部，9.78%位于裁剪外部，提示还需检查裁剪内的预测响应；该比例仅适用于此失败子集。')

    node=anchor('按粗框确定邻线后，使用精修框时')
    paired=csvrows('08_paired_geometry.csv')
    node=table(node,'表3 粗框与精修框引导分割的配对结果',['指标','粗框/%','精修框/%','变化/百分点'],[[r['指标'].replace(' (%)','').replace('固定粗框邻线队列互换正确率','固定队列互换成功率'),*[f'{float(r[k]):.3f}' for k in ('粗框','精修框','变化（百分点）')]] for r in paired],'使用同一axes分割器与候选身份，匹配实例共3140个。互换采用固定粗框定义的346次试验，与表2的342次队列不同，不直接横向合并。分割器使用精修框训练，因此结果也包含对训练输入的适配影响。')
    figure(node,ASSETS/'figures/08_paired_geometry_changes.png','图8 候选几何变化与下游掩码变化的配对分析','散点保留全部3140个匹配实例，右侧计数展示框IoU与掩码IoU变化的组合。掩码IoU提高、不变和下降的实例分别为1818、86和1236个；整体Mask mAP提高1.254个百分点，但框IoU提高并不保证每个实例的掩码同步改善。')

    node=anchor('初步方案将按尺寸对候选分组')
    figure(node,ASSETS/'方法图导出/method_4.png','图9 面向边缘设备的检测与实例分割协同部署方案','图9表示拟在Jetson Orin NX上实施的模块划分及数据传递。候选尺寸分组、像素预算调度和加速执行仍需板端实现与验证，服务器端基线不能替代板端吞吐率、功耗及持续运行评价。')
    node=anchor('后续将把检测与分割流程迁移到 Jetson 平台')
    node=para(node,'3.3 服务器端完整流程效率基线')
    node=para(node,'为给后续板端优化提供参照，在RTX 4090上测量在线检测、精修、裁剪与提示、分割及掩码恢复的完整流程。输入为1695张640×640验证图块，采用FP32张量、单图块及逐候选分割，每轮预热50次后测量，共重复三轮。计时包含图像读取，排除模型加载、预热、AP计算、结果写盘和相机采集显示。')
    node=para(node,'三轮5085次测量的平均时延为37.59毫秒，中位数为32.24毫秒，P95为61.82毫秒，P99为110.92毫秒。平均时延倒数为26.60图块/秒，仅表示该串行测量口径下的换算速度。三轮均值为39.13、36.73和36.91毫秒，极差相对均值为6.40%，尚高于预设5%的稳定性门限。在线候选、重复输出及分阶段计时输出核对均未发现不一致。')
    node=figure(node,ASSETS/'figures/E01_stage_latency.png','图10 服务器完整推理流程的分阶段耗时','采用独立同步计时轮分解各阶段平均耗时，检测与旋转NMS约19.34毫秒、尺度精修2.08毫秒、裁剪与提示4.28毫秒、分割推理及回传7.13毫秒。分阶段统计用于定位开销，不将各阶段分位数相加替代端到端分位数。')
    figure(node,ASSETS/'figures/E02_workload_latency.png','图11 候选数量、裁剪面积与完整流程时延','每个点代表一张图块的三轮平均时延，共1695个点。包含20根及以上候选的17张图块，其组内平均时延约230.47毫秒，说明密集场景存在明显长尾。后续应评估逐候选调用、提示构造和批次调度的开销；目前尚未据此证明某项加速方案已经有效。')

    # Verify insertion-only semantics before replacing the user's file.
    after_nodes=list(doc.element.body.iter(qn('w:p')))
    for p,t in zip(originals,old_text):
        assert p in after_nodes and Paragraph(p,doc).text==t
    tmp=REPORT.with_name('中期-初稿_插图临时.docx')
    doc.save(tmp)
    checked=Document(tmp)
    assert len(checked.inline_shapes)==len(Document(backup).inline_shapes)+11
    tmp.replace(REPORT)
    with (ASSETS/'tables/03_prompt_results_with_swap.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['prompt','mask_map_pct','neighbor_recall_pct','neighbor_leakage_pct','swap_success_count','swap_success_pct']);writer.writerows(metrics)
    record={'report':str(REPORT),'backup':str(backup),'added_figures':11,'added_tables':3,'original_paragraphs_preserved':True,'omitted':['实验设置表（原清单1）','精修倍率图（原清单10）'],'visual_review':False,'changes':changes,'source_sha256':sources}
    (ASSETS/'本次插图记录.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('report','backup','added_figures','added_tables','original_paragraphs_preserved')},ensure_ascii=False))


if __name__=='__main__':
    main()

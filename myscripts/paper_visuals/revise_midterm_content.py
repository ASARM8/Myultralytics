"""Revise report narrative without changing figures, numeric tables or styles."""
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime
from copy import deepcopy
import json
import shutil
from lxml import etree as E

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'mydocs/中期报告'
PATH=BASE/'中期-初稿.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
NS={'w':W[1:-1]}


def main():
    with ZipFile(PATH) as z:
        original={i.filename:z.read(i.filename) for i in z.infolist()}
        infos=z.infolist()
    root=E.fromstring(original['word/document.xml'])
    paragraphs=list(root.iter(W+'p'))
    def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=NS))
    def find(prefix):
        found=[p for p in paragraphs if text(p).startswith(prefix)]
        assert len(found)==1,(prefix,len(found))
        return found[0]
    changed=[]
    def replace(prefix,new):
        p=find(prefix)
        old=text(p)
        assert not p.xpath('.//w:drawing|.//w:object|.//m:oMath',namespaces={**NS,'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'})
        run=p.find(W+'r')
        style=deepcopy(run.find(W+'rPr')) if run is not None and run.find(W+'rPr') is not None else None
        for child in list(p):
            if child.tag!=W+'pPr':p.remove(child)
        r=E.SubElement(p,W+'r')
        if style is not None:r.append(style)
        E.SubElement(r,W+'t').text=new
        changed.append({'before':old,'after':new})
        return p
    def remove(prefix):
        p=find(prefix);changed.append({'before':text(p),'after':'[删除重复描述]'});p.getparent().remove(p)
    def append(after,content):
        p=E.Element(W+'p',nsmap=after.nsmap)
        pp=after.find(W+'pPr')
        if pp is not None:p.append(deepcopy(pp))
        r=E.SubElement(p,W+'r');E.SubElement(r,W+'t').text=content
        after.addnext(p);changed.append({'before':'[新增]','after':content})
        return p

    # Keep the finite-axis mechanism and contribution-boundary text untouched.
    protected=[E.tostring(p) for p in paragraphs if text(p).startswith((
        '每个旋转框的主轴均表示为具有两个端点的有限线段。',
        '用当前目标构造独立的目标轴线通道',
        '当指定的分割对象改变时，同一候选的轴线可以'))]
    for prefix in (
        '图中展示从输入图像、旋转框检测和尺度精修',
        '图4展示单个候选的定向裁剪、目标提示、邻线提示'):
        remove(prefix)
    replace('整体处理流程如图1所示。','整体流程如图1所示。检测与精修首先提供候选几何，分割网络再依据定向条带及角色提示输出实例掩码。候选编号与坐标变换贯穿各阶段，使目标级定位与像素级输出保持对应。')
    replace('图2将训练时的范围约束和推理时的尺度更新分别表示。','图2将范围感知样本分配与候选尺度精修连接起来：前者改善候选形成所依赖的训练分配，后者在固定实例身份下更新长短边，使几何优化能够继续传递到分割裁剪与提示构造。')
    if any(text(p).startswith('在当前验证集上，采用回归范围感知配置后') for p in paragraphs):
        replace('在当前验证集上，采用回归范围感知配置后','表1给出了训练分配与尺度精修的逐阶段收益。回归范围感知配置首先提升候选定位质量，尺度精修随后进一步提高检测精度，形成从训练样本选择到候选几何修正的连续优化。')
    replace('表1中CA为扩展回归区间与回归范围感知分配的联合配置','CA配置将OBB mAP50–95由40.840%提高到45.414%，增加4.574个百分点；加入尺度精修后达到49.581%，再提高4.167个百分点。Precision与Recall分别达到84.541%和81.809%，相对CA同时提高，说明候选尺度修正改善了检测质量。表中CA采用扩展回归区间与范围感知分配的联合设置，精修残差倍率为0.42。')
    replace('绿色为真实旋转框，橙色为预测框；','图3中绿色为真值、橙色为预测框，每行采用相同局部窗口，ΔIoU按完整图块的匹配结果计算。前两例展示尺度调整后预测边界向真实边界靠近的过程，第三例展示修正方向与目标尺度不一致时的定位退化。')

    replace('对于长度不同的裁剪区域，训练时按填充后的像素预算组织批次','定向条带的长度和宽度随候选变化，批处理时需要将它们填充到共同尺寸。普通Group Normalization会将填充区一并纳入空间均值与方差计算，使同一条带的归一化结果受到批次填充尺寸影响；卷积与归一化在填充区产生的响应还会经后续卷积传播到有效边缘。因此，本文采用有效区域约束，只在原始条带范围内计算归一化统计量，并在每个卷积块后清除填充区响应，使批处理保持逐条计算的有效边界。')
    append(find('定向条带的长度和宽度随候选变化'), '训练按填充后的像素预算组织批次，结合多进程预取和混合精度提高处理效率。批处理与独立计算的输出及梯度一致性测试已通过，从而在保留原始条带计算结果的同时完成变长实例的批量训练。')

    # Evaluation definitions belong before results, rather than in threshold analysis.
    replace('2.4 不同邻线表示的对照设计','2.4 邻线表示对照与评价设置')
    definition=find('依据真实实例的可见像素确定邻线关系')
    replace('依据真实实例的可见像素确定邻线关系','评价从实例选择、像素质量和邻线对完整分割三个层面展开。角色互换在同一RGB裁剪中交换目标与邻线提示，检查两次输出是否分别更接近指定实例。整体分割采用COCO多IoU阈值的Mask mAP50–95[35]，匹配实例另统计掩码IoU、目标像素召回和邻线泄漏。')
    neighbor=append(definition,'真值邻线关系由可见实例像素确定：沿用短轴两侧各12像素、长轴两端各约4像素的条带范围，若其中包含另一实例独有的可见像素，就将两者记为邻线，每个无序对只计一次；主轴夹角不超过15°归入近平行子集。验证集共270对真值邻线，所有方法和阈值共用该集合，检测未覆盖的实例也进入分母。两根实例掩码均达到指定IoU时，该邻线对计为成功。')
    h25=find('2.5 整体分割质量与角色互换实验')
    h25.addprevious(definition);h25.addprevious(neighbor)
    replace('2.5 整体分割质量与角色互换实验','2.5 指定实例选择能力与整体分割结果')
    replace('整体分割质量参考 COCO 的多阈值评价方法','角色化轴线提示首先改善了指定实例选择能力。342次互换试验中，axes成功229次，成功率为66.959%，高于target的36.550%、points的61.404%和boxes的57.895%，分别增加30.409、5.556和9.064个百分点，结果见表2。')
    replace('在验证集的默认掩码阈值下，四种提示设置的整体掩码精度接近','互换前后保持RGB与裁剪位置相同，分割对象随提示角色改变。axes取得最多的双向选择成功次数，说明网络能够利用目标与邻线的角色信息确定当前实例，而非始终输出裁剪中同一根外观突出的电线。')
    remove('角色互换试验固定 RGB 图像和裁剪区域')
    remove('角色互换试验根据相对 IoU 判断模型选中了哪个实例')
    replace('掩码二值化阈值为0.5。Mask mAP采用完整验证集','表2采用掩码阈值0.5；Mask mAP统计完整验证集，召回与泄漏统计386个邻线暴露匹配实例，互换统计342次有方向试验。互换成功按两次输出与指定实例的相对IoU判定。')
    replace('表2中，axes的互换成功率为66.959%','axes在提升目标选择能力的同时取得四组中最高的整体Mask mAP50–95，为46.961%；target、points和boxes分别为46.344%、46.893%和46.881%。这表明角色化邻线表示主要强化实例归属，整体掩码质量则保持在同一水平。常规推理中的双实例完整分割结果如图5所示。')
    # Move the neighbor-pair figure from threshold section to the result section.
    cap6=find('图6 不同掩码阈值下的双实例分割成功邻线对数')
    pic6=cap6.getprevious()
    assert pic6.xpath('.//w:drawing',namespaces=NS)
    description6=cap6.getnext()
    block=[pic6,cap6,description6]
    h26=find('2.6 真值邻线对与多阈值评价')
    for p in block:h26.addprevious(p)
    replace('图6 不同掩码阈值下的双实例分割成功邻线对数','图5 不同掩码阈值下的双实例分割成功邻线对数')
    replace('以270对真实邻线为共同分母','图5以270对真实邻线为共同分母。在掩码阈值0.5、IoU门限0.5时，target成功130对，其余三组均为129对；阈值降至0.2时，axes增加至158对，target为160对。目标选择能力的提升最鲜明地体现在互换试验中，常规推理的双实例成功数量则处于相近水平。')

    replace('2.6 真值邻线对与多阈值评价','2.6 掩码阈值下的目标召回与邻线泄漏权衡')
    replace('验证集中的真值邻线对以近平行电线为主','阈值决定前景概率转化为可见像素的工作点。降低阈值能够保留更多低响应目标像素，同时增加邻线及背景进入掩码的机会。为比较不同提示的像素归属能力，对每个候选生成一次概率图，扫描0.1至0.5的二值化阈值，并在共同的386个邻线暴露实例上联合考察目标召回与邻线泄漏，结果如图6所示。')
    replace('在完整验证集上，每个候选只生成一次概率图','在目标召回约77%—78%的实测工作区间，axes的召回为77.492%、泄漏为2.980%；target对应77.084%和4.218%，points对应77.964%和3.515%。axes保留了相近比例的目标像素，同时将泄漏分别降低1.237和0.534个百分点，体现出角色化轴线提示对邻线干扰的抑制作用。')
    replace('在轴线提示组中，降低掩码阈值后','boxes在这一工作区间的召回为77.775%、泄漏为2.979%，与axes接近。两者均能提供覆盖邻线的空间条件；axes进一步通过连续主轴表达方向与沿线范围，在角色互换中获得更高的指定实例选择成功率。连续几何条件改善的是实例归属引导，不同指标从不同侧面展示这一作用。')
    replace('在包含邻线的匹配实例上，与仅使用目标提示相比','axes自身的阈值扫描展示了清晰的目标保留与干扰抑制关系：阈值由0.5降至0.2，目标召回由73.690%提高至85.337%，邻线泄漏由2.753%增加至3.692%。因此，可以依据任务对漏检和误分的侧重选择运行阈值；完整验证集的最高采样Mask mAP出现在0.4，axes为47.601%，兼顾目标像素恢复与干扰控制。')
    replace('在目标像素召回率相近的实测工作点','图7进一步展示不同阈值的作用。第一例的低响应目标像素得到恢复，第二例仍存在明显缺失，第三例则因扩大前景范围引入更多误分。37个失败实例中90.22%的遗漏像素位于裁剪内部，说明后续质量提升的重点是已有图像范围内的线体响应与像素归属。')
    replace('图5 目标召回、邻线泄漏与掩码阈值的关系','图6 目标召回、邻线泄漏与掩码阈值的关系')
    replace('左图统计386个邻线暴露实例','左图为386个邻线暴露实例上的召回—泄漏工作点，右图为1695张验证图块上的整体Mask mAP。曲线点对应掩码阈值0.1至0.5。')
    replace('同一axes概率图分别按0.5和0.2二值化','图7使用同一axes概率图按0.5和0.2二值化。每行展示相同256×256局部区域，IoU按完整640×640图块中的实例计算；三行依次对应目标恢复、持续漏分和误分扩大。')

    replace('配对实验固定 axes 组的分割权重','为检验检测几何对分割的传递作用，固定axes权重、候选身份与检测分数，分别依据粗框和精修框进行裁剪及提示构造。两组预测逐实例对应，配对结果见表3。')
    replace('对同一批候选，使用精修框时的整体','精修框使整体Mask mAP50–95由45.707%提高到46.961%，增加1.254个百分点；平均框IoU由71.710%升至73.893%，平均掩码IoU由75.741%升至76.159%。检测阶段的尺度优化由此转化为下游分割收益，建立了两个创新点之间的实验联系。')
    replace('按粗框确定邻线后，使用精修框时','固定邻线队列的互换成功率由69.075%降至65.896%，表明裁剪及提示随尺度更新发生变化，整体分割收益与角色切换结果呈现不同响应。该结果为后续优化目标覆盖和邻线条件的协同关系提供了方向。')
    replace('使用同一axes分割器与候选身份，匹配实例共3140个','表3统计3140个固定匹配实例，互换采用粗框定义的346次固定队列。')
    replace('散点保留全部3140个匹配实例','图8展示逐实例变化：1818个实例的掩码IoU提高、86个不变、1236个下降。框与掩码的配对分布将整体收益展开到实例层面，为定位尺度调整后受益及退化的目标提供依据。')
    deployment=text(find('图9表示拟在Jetson Orin NX上实施的模块划分及数据传递'))
    plan=deployment[deployment.index('后续将把检测与分割流程迁移到 Jetson 平台'):] if '后续将把检测与分割流程迁移到 Jetson 平台' in deployment else ''
    replace('图9表示拟在Jetson Orin NX上实施的模块划分及数据传递','图9给出Jetson Orin NX部署中的模块分工、候选调度及数据传递。'+plan)
    replace('采用独立同步计时轮分解各阶段平均耗时','图10采用独立同步计时轮分解阶段开销。检测与旋转NMS为19.34毫秒，约占51.4%；分割推理及回传为7.13毫秒，约占19.0%；裁剪与提示为4.28毫秒，约占11.4%；精修为2.08毫秒，约占5.5%。检测计算与逐候选分割构成主要优化对象。')
    replace('每个点代表一张图块的三轮平均时延','图11中每点为一张图块的三轮平均时延。20根及以上候选的17张图块，组内平均时延为230.47毫秒，显示候选密集场景的负载集中增长。部署调度将据此按候选尺寸和像素预算组织批次，减少逐条调用与填充计算开销。')

    # Verify that the only archive part changed is the requested report text.
    for blob in protected:assert any(E.tostring(p)==blob for p in root.iter(W+'p'))
    assert len(root.xpath('.//w:drawing',namespaces=NS))==len(E.fromstring(original['word/document.xml']).xpath('.//w:drawing',namespaces=NS))
    assert len(root.xpath('.//w:tbl',namespaces=NS))==len(E.fromstring(original['word/document.xml']).xpath('.//w:tbl',namespaces=NS))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=BASE/f'中期-初稿_内容优化前备份_{stamp}.docx'
    shutil.copy2(PATH,backup)
    temp=BASE/'中期-初稿_内容优化临时.docx'
    with ZipFile(temp,'w') as z:
        for i in infos:z.writestr(i,E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True) if i.filename=='word/document.xml' else original[i.filename])
    with ZipFile(temp) as z:
        assert z.testzip() is None
        for k,v in original.items():
            if k!='word/document.xml':assert z.read(k)==v
    temp.replace(PATH)
    record={'backup':str(backup),'changes':changed,'format_review':False,'problem1_mechanism_text_preserved':True,'new_data_required':False,'only_modified_package_part':'word/document.xml'}
    (BASE/'图表补充_20261114/本轮内容优化记录.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'report':str(PATH),'backup':str(backup),'text_changes':len(changed),'images_and_tables_preserved':True},ensure_ascii=False))


if __name__=='__main__':main()

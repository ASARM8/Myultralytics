from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.shared import Pt, RGBColor

from myscripts.paper_submission.build_target_branches import TARGETS


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "mydocs" / "创新点一" / "创新点一_覆盖能力感知与候选框级几何精修方法_数据版.docx"
DEFAULT_OUTPUT = ROOT / "mydocs" / "创新点一" / "投稿版本"


ZH_TARGETS = {
    "ivc": {
        "title": "面向无人机图像中电线旋转检测的覆盖能力感知分配与身份保持几何精修",
        "header": "面向无人机图像的电线旋转检测与几何精修",
        "abstract": (
            "低空无人机图像中的电线短轴纹理证据少、长轴跨越范围大且方向连续变化，容易暴露特征金字塔"
            "旋转检测器中的几何不相容问题：任务对齐分配可能选中语义上合适、但有限分布式回归范围无法"
            "覆盖目标全部边界的正样本。本文提出 YOLO11-OBB 的覆盖能力感知扩展。该方法在目标局部坐标系"
            "中计算候选点到边界的最大需求距离，并在任务对齐排序前过滤几何不可达候选；reg_max=32 用于"
            "为不同金字塔层提供必要的距离表达支撑。为修正剩余的高 IoU 尺度误差，本文进一步围绕 post-NMS"
            "候选框从 P2/P3 采样旋转对齐区域，仅预测短边和长边的有界对数尺度残差，同时保持中心、角度、"
            "置信度、候选数量和 NMS 身份不变。在 imgsz=640 的固定 TTPLA 验证协议下，候选框精修将"
            " mAP50-95 从 0.4541 提升至 0.5625，AP75 从 0.4398 提升至 0.5588；AP90 提高 0.0157，而"
            " AP95 下降 0.0018。严格恒等对照、三种无重训练评估路径和 proposal 匹配分析表明，收益来自"
            "学习到的几何校正：匹配候选平均 IoU 提高 0.0466，63.41% 的匹配候选得到改善。该结果支持"
            "方法对电线旋转定位具有明确但有边界的正向作用；多种子、复杂度和最终测试集证据仍待补充。"
        ),
        "keywords": "电线检测；无人机图像；旋转目标检测；正样本分配；分布式回归；候选框精修",
        "note": (
            "内部投稿分支说明（投稿前删除）：本稿是 Image and Vision Computing 的后续主稿。方法冻结为"
            " Coverage-Aware Assignment + reg_max=32 表达支撑 + 身份保持的候选框级几何精修。本轮不训练"
            "模型，只使用已经审计的结果；统一 Baseline、H1/H2、复杂度、多种子、定性结果和最终测试集数据"
            "在未实测前保留为空，不用预期值替代。"
        ),
        "framing": (
            "从图像与视觉计算角度看，本文关注的核心不是无人机飞控，而是语义候选排序、有限几何表达与"
            " proposal 级定位之间的不匹配。低空无人机电线场景使这一问题更突出；本文将其作为可受控验证的"
            "旋转目标检测问题处理，不把检测结果扩展表述为闭环避障能力。"
        ),
        "limitation": (
            "面向 Image and Vision Computing，当前主要风险是证据完整性和泛化范围。统一原始 Baseline、"
            "覆盖溢出机制统计、完整计算代价、多种子和冻结测试集仍需补齐；这些属于冻结方法后的证据补全，"
            "不应据此继续更改当前算法。"
        ),
        "conclusion": (
            "现有证据支持覆盖能力感知分配与身份保持候选框精修对电线旋转定位具有正向作用，但结论应保持"
            "在单数据集、单正式种子和当前验证协议范围内，直至剩余证据补齐。"
        ),
    },
    "unmanned": {
        "title": "面向无人机低空感知的电线障碍物覆盖能力感知检测与候选框级几何精修",
        "header": "面向无人机低空感知的电线障碍物检测",
        "abstract": (
            "低空无人机在巡检、测绘和近基础设施飞行过程中需要可靠识别悬空电线。此类碰撞相关障碍物"
            "短边像素少、长边跨度大且方向任意，使特征金字塔旋转检测器面临正样本分配与有限分布式回归"
            "范围不匹配的问题：语义上合适的候选点未必能够完整覆盖目标四条边界。本文以 YOLO11-OBB 为"
            "基础构建电线感知模块。首先，在旋转真实框局部坐标系中计算候选点完整覆盖目标所需的最大边界"
            "距离，并在任务对齐排序前过滤几何不可达候选；同时将 reg_max 扩展为 32，为细长目标提供足够"
            "的距离表达支撑。随后，以 post-NMS 粗框为对象，从 P2/P3 提取方向对齐的局部特征，仅预测短边"
            "和长边的有界对数尺度残差，并保持中心、角度、类别置信度与候选框身份不变。在固定 imgsz=640、"
            "FP32、batch=8 的验证协议下，候选框级精修将 mAP50-95 从 0.4541 提升至 0.5625，AP75 从 0.4398"
            "提升至 0.5588；AP90 提高 0.0157，AP95 轻微下降 0.0018。恒等对照和 proposal 级分析显示，"
            "匹配框平均 IoU 增量为 0.0466，63.41% 的匹配候选得到改善。上述结果支持该方法对低空无人机"
            "电线感知具有明确但有边界的几何改进作用；本文尚未验证测距、闭环避障或机载实时部署。"
        ),
        "keywords": "无人机；低空感知；电线障碍物检测；旋转目标检测；正样本分配；几何精修",
        "note": (
            "内部投稿分支说明（投稿前删除）：目标期刊为 Unmanned Systems。正文按“无人机低空视觉感知模块”"
            "定位，投稿前须补完整 detector+NMS+Refiner 时延、参数量、FLOPs、显存和 proposal 数量相关耗时；"
            "最好增加边缘设备测试。没有闭环飞行实验时不得写成自主避障系统。"
        ),
        "framing": (
            "从无人系统角度看，本文检测器承担的是低空飞行安全链路中的视觉前端功能：将低可见度电线转换为"
            "保留方向和空间范围的旋转框输出，供后续风险评估或路径规划模块使用。现有实验只评价图像空间中的"
            "识别与几何定位，不涉及真实距离估计、飞行器动力学、航迹生成或闭环控制。"
        ),
        "limitation": (
            "面向无人机部署，当前结果仍属于离线图像感知验证。正式投稿前需测量包含解码、NMS 和旋转 ROI "
            "采样在内的完整时延；在缺少目标机载平台测试时，不应使用“实时机载”或“部署就绪”等表述。"
        ),
        "conclusion": (
            "本文输出可作为低空无人机安全感知链路中的图像空间电线 OBB，但与测距、风险判断、路径规划和"
            "飞行控制的系统集成仍需后续研究。"
        ),
    },
    "jars": {
        "title": "面向无人机低空遥感图像的电线障碍物覆盖能力感知检测与候选框级几何精修",
        "header": "面向无人机低空遥感图像的电线障碍物检测",
        "abstract": (
            "无人机低空遥感图像中的电线短边通常只有少量像素，却沿任意方向跨越较大空间范围。这种几何特征"
            "会造成特征金字塔正样本分配与分布式边界回归之间的不匹配：候选点虽然具有较高任务对齐得分，"
            "但其所在层级的有限距离表示范围可能无法覆盖完整旋转目标。本文以 YOLO11-OBB 为基础提出覆盖"
            "能力感知的电线旋转检测方法。首先，在目标局部坐标系中计算候选点覆盖四条边界所需的最大距离，"
            "并在任务对齐分配前剔除超出当前层表达范围的候选；reg_max=32 作为细长目标长距离回归的表达"
            "支撑。其次，候选框级几何精修器围绕 post-NMS 粗框采样方向对齐的 P2/P3 特征，只学习短边与长边"
            "的连续尺度残差，保持中心、角度、类别分数和候选身份不变。在固定 640 像素 FP32 验证协议下，"
            "mAP50-95 由 0.4541 提升至 0.5625，AP75 由 0.4398 提升至 0.5588；AP90 提高 0.0157，而 AP95"
            "下降 0.0018。恒等对照与 proposal 级 IoU 统计支持收益来自实例相关的尺度校正。本文为无人机"
            "低空巡检和环境测绘中的电线大长宽比旋转定位提供了一种几何建模方案。"
        ),
        "keywords": "无人机遥感；电线障碍物检测；旋转目标检测；覆盖能力感知；分布式回归；几何精修",
        "note": (
            "内部投稿分支说明（投稿前删除）：目标期刊为 Journal of Applied Remote Sensing。正文突出 UAV "
            "低空遥感、大长宽比 OBB 和空间定位价值；投稿前须补统一 Baseline/CA、H1/H2、复杂度/FPS 和"
            "全英文定性图，第二公开遥感 OBB 数据集为加分项。"
        ),
        "framing": (
            "从无人机遥感角度看，关键不仅是识别一条细线，还要恢复稳定且紧致的旋转空间范围。电线短边可能"
            "接近传感器采样极限，而长边跨越多个特征单元；其旋转框质量直接影响巡检目标映射、走廊分析、"
            "植被净空评估和后续空间测量。"
        ),
        "limitation": (
            "遥感泛化方面，当前正式证据集中于 TTPLA 低空航拍域。若要把结论扩展到更广泛的大长宽比旋转"
            "目标，应增加第二公开 OBB 数据集或跨域测试；在此之前，结论限定于当前数据和采集条件。"
        ),
        "conclusion": (
            "该方法面向无人机低空遥感图像中的电线几何定位，其对其他大长宽比目标的泛化能力仍需额外数据集验证。"
        ),
    },
    "jei": {
        "title": "面向无人机低空成像感知的电线障碍物覆盖能力感知检测与候选框级几何精修",
        "header": "面向无人机低空成像感知的电线障碍物检测",
        "abstract": (
            "低空无人机图像中的电线具有长轴跨度大、短轴纹理证据少和方向连续变化等特征，容易暴露旋转检测器"
            "中的结构性定位问题：任务对齐分配可能选择离散距离分布无法完整覆盖目标的候选点。本文提出"
            " YOLO11-OBB 的几何感知扩展。Coverage-Aware Assignment 在目标对齐坐标系中计算每个候选点"
            "所需的最大边界距离，并抑制超出特征层表达范围的候选；reg_max=32 被视为覆盖分配的表达支撑，"
            "而非独立创新。随后，候选框级精修器围绕 post-NMS 粗框提取旋转 P2/P3 区域，仅预测短边和长边"
            "的对数尺度残差。中心、角度、类别置信度和 NMS 结果均保持不变，从而可进行严格恒等对照。在"
            " imgsz=640、FP32、batch=8 的固定协议下，mAP50-95 从 0.4541 提升至 0.5625，AP75 从 0.4398"
            "提升至 0.5588。proposal 匹配结果显示平均 IoU 增量为 0.0466，63.41% 的匹配框得到改善；"
            "AP95 下降 0.0018，说明该方法获得了明确但非所有阈值一致的定位改善。"
        ),
        "keywords": "无人机成像；电线障碍物检测；旋转框；正样本分配；分布式回归；候选框精修",
        "note": (
            "内部投稿分支说明（投稿前删除）：目标期刊为 Journal of Electronic Imaging。正文突出无人机"
            "成像定位链、恒等对照、proposal 诊断和复现性；投稿前须补统一 Baseline、复杂度/完整时延和"
            "全英文可视化，单种子结果不得写成统计显著。"
        ),
        "framing": (
            "从电子成像与定位角度看，本文贡献是对旋转框定位链的受控干预：将有限距离编码、特征金字塔分配"
            "和候选框局部图像采样联系起来，并通过同权重恒等模式排除置信度重标定、二次抑制和结果回写对"
            "精度变化的干扰。"
        ),
        "limitation": (
            "面向成像系统部署，仍需在目标硬件上测量内存和完整时延。只有在 detector、解码、NMS、proposal "
            "采样与精修的整体吞吐量经过验证后，才能使用实时性表述。"
        ),
        "conclusion": (
            "该结果证明了无人机低空成像中的电线旋转框可通过受控候选级尺度校正获得改善，但完整系统速度与"
            "跨硬件复现仍需补充。"
        ),
    },
}


def _set_run_font(run, cn: str = "宋体", latin: str = "Times New Roman", size: float = 10.5, bold=None) -> None:
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), cn)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def _replace_paragraph(paragraph: Paragraph, text: str, *, cn: str = "宋体", size: float = 10.5, bold=False) -> None:
    paragraph.clear()
    run = paragraph.add_run(text)
    _set_run_font(run, cn=cn, size=size, bold=bold)


def _replace_labelled(paragraph: Paragraph, label: str, text: str) -> None:
    paragraph.clear()
    label_run = paragraph.add_run(label)
    _set_run_font(label_run, cn="黑体", size=10.5, bold=True)
    body_run = paragraph.add_run(text)
    _set_run_font(body_run, cn="宋体", size=10.5)
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def _insert_after(paragraph: Paragraph, text: str, *, note: bool = False) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    p = Paragraph(new_p, paragraph._parent)
    if note:
        p_pr = p._p.get_or_add_pPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "FFF2CC")
        p_pr.append(shd)
        run = p.add_run(text)
        _set_run_font(run, cn="黑体", size=9.5, bold=True)
        run.font.color.rgb = RGBColor(156, 87, 0)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
    else:
        try:
            p.style = "Body Text"
        except KeyError:
            pass
        run = p.add_run(text)
        _set_run_font(run, cn="宋体", size=10.5)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Pt(21)
        p.paragraph_format.space_after = Pt(3)
    return p


def _find_paragraph(doc: Document, prefix: str) -> Paragraph:
    for p in doc.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    raise ValueError(f"paragraph not found: {prefix}")


def build(target_key: str, output_dir: Path) -> Path:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    meta = ZH_TARGETS[target_key]
    en_meta = TARGETS[target_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{en_meta['file_stem']}_投稿预备分支_中文版.docx"
    shutil.copy2(SOURCE, out)

    doc = Document(out)
    nonempty = [p for p in doc.paragraphs if p.text.strip()]
    _replace_paragraph(nonempty[0], meta["title"], cn="黑体", size=18, bold=True)
    nonempty[0].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Keep the running header consistent with the target-specific body title.
    # Some source templates share the same header part across sections, so track
    # part identities to avoid replacing the same paragraph repeatedly.
    seen_header_parts: set[int] = set()
    for section in doc.sections:
        for header in (section.header, section.first_page_header, section.even_page_header):
            header_part_id = id(header.part)
            if header_part_id in seen_header_parts:
                continue
            seen_header_parts.add(header_part_id)
            for paragraph in header.paragraphs:
                if paragraph.text.strip():
                    _replace_paragraph(paragraph, meta["header"], cn="宋体", size=8.5)
                    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    author = _find_paragraph(doc, "作者：")
    _insert_after(author, meta["note"], note=True)

    abstract = _find_paragraph(doc, "摘要：")
    _replace_labelled(abstract, "摘要：", meta["abstract"])
    keywords = _find_paragraph(doc, "关键词：")
    _replace_labelled(keywords, "关键词：", meta["keywords"])

    english_abstract = _find_paragraph(doc, "Power lines")
    _replace_paragraph(english_abstract, en_meta["abstract"], cn="Times New Roman", size=9.5)
    english_abstract.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    english_keywords = _find_paragraph(doc, "Key words:")
    _replace_paragraph(english_keywords, f"Key words: {en_meta['keywords']}", cn="Times New Roman", size=9.5)

    intro_first = _find_paragraph(doc, "电线、拉线和细杆等低空障碍物")
    framing_anchor = _insert_after(intro_first, meta["framing"])
    if target_key == "ivc":
        _insert_after(
            framing_anchor,
            "与近期 Image and Vision Computing 论文相比，Bai 等侧重自注意引导、全局特征融合与小目标分配，Sang 等侧重环境自适应上下文和快速检测，Chaurasia 与 Patro 侧重通道—空间注意及旋转角分类；Rong 等则从方向特征增强切入电线检测。本文不直接横向比较跨数据集 AP，而是把差异限定为算法问题：正样本是否在有限距离分布下几何可表示，以及 post-NMS 旋转候选能否在保持身份与置信度不变的前提下获得局部尺度校正。",
        )

    limitation = _find_paragraph(doc, "本文仍存在三方面局限")
    _insert_after(limitation, meta["limitation"])

    conclusion_last = _find_paragraph(doc, "在固定 FP32、batch=8、imgsz=640")
    _insert_after(conclusion_last, meta["conclusion"])

    if target_key == "ivc":
        identity_result = _find_paragraph(doc, "表9表明，coarse 与 identity")
        _insert_after(
            identity_result,
            "无重训练评估路径复核如下：FP32、batch=8 时 coarse/refined 为 0.454137/0.562504，增量 0.108368；FP32、batch=1 时为 0.454151/0.562460，增量 0.108310；AMP、batch=8 时为 0.453562/0.562017，增量 0.108455。三条路径的增量极差仅 0.000146，说明主提升不依赖单一 batch 或数值精度设置。该对照复用同一 checkpoint，只属于评估链稳健性测试，不能视为独立种子或统计显著性证据。",
        )

    props = doc.core_properties
    props.title = meta["title"]
    props.subject = f"{en_meta['journal']} 中文投稿审阅分支"
    props.keywords = meta["keywords"]
    props.comments = "内部中文审阅分支；投稿前删除黄色说明并按目标期刊模板处理。"
    doc.save(out)
    # Remove stale font-table declarations that can make Word report a
    # non-used Japanese fallback (e.g. MS Gothic) in the font inspector.
    from lxml import etree
    from zipfile import ZIP_DEFLATED, ZipFile

    temp = out.with_suffix(".fontfix.docx")
    with ZipFile(out, "r") as src, ZipFile(temp, "w", ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/fontTable.xml":
                root = OxmlElement("w:fonts")
                for name in ("Times New Roman", "宋体", "黑体", "Cambria Math"):
                    font = OxmlElement("w:font")
                    font.set(qn("w:name"), name)
                    root.append(font)
                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
            dst.writestr(item, data)
    temp.replace(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chinese review counterparts for target-specific journal branches.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", choices=[*ZH_TARGETS, "all"], default="all")
    args = parser.parse_args()
    keys = ZH_TARGETS if args.target == "all" else [args.target]
    for key in keys:
        print(build(key, args.output_dir))


if __name__ == "__main__":
    main()

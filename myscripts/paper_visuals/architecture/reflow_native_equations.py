"""Merge fragmented PowerPoint equations into aligned, full-line OMML objects.

The source decks already contain editable Office Math objects created by Word.
This script reuses those native objects, replaces their OMML payload with a
complete expression, removes superseded fragments, and normalizes the formula
font size and bounding box. Only ``ppt/slides/slide1.xml`` is changed.
"""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
EMU_PER_POINT = 12700


@dataclass(frozen=True)
class FormulaSpec:
    """One complete equation and its placement relative to an anchor shape."""

    keep: str
    anchor: str
    tokens: tuple[tuple[str, ...], ...]
    font_size: float
    color: str
    left: float
    top: float
    width: float
    height: float
    bold: bool = False


def qn(prefix: str, local: str) -> str:
    """Return a Clark-notation qualified name."""
    return f"{{{NS[prefix]}}}{local}"


def pt(value: float) -> int:
    """Convert PowerPoint points to integer EMUs."""
    return round(value * EMU_PER_POINT)


def direct_shapes(root: ET._Element) -> dict[str, ET._Element]:
    """Return ordinary top-level slide shapes by name, excluding fallbacks."""
    tree = root.find(".//p:spTree", NS)
    if tree is None:
        raise ValueError("Slide shape tree was not found")
    result: dict[str, ET._Element] = {}
    for shape in tree.findall("p:sp", NS):
        props = shape.find("p:nvSpPr/p:cNvPr", NS)
        if props is not None and props.get("name"):
            result[props.get("name")] = shape
    return result


def equation_objects(root: ET._Element) -> dict[str, ET._Element]:
    """Return top-level AlternateContent wrappers containing native equations."""
    tree = root.find(".//p:spTree", NS)
    if tree is None:
        raise ValueError("Slide shape tree was not found")
    result: dict[str, ET._Element] = {}
    for alternate in tree.findall("mc:AlternateContent", NS):
        props = alternate.find("mc:Choice/p:sp/p:nvSpPr/p:cNvPr", NS)
        if props is not None and props.get("name"):
            result[props.get("name")] = alternate
    return result


def shape_geometry(shape: ET._Element) -> tuple[int, int, int, int]:
    """Return ``left, top, width, height`` in EMUs for an ordinary shape."""
    transform = shape.find("p:spPr/a:xfrm", NS)
    if transform is None:
        raise ValueError("Shape transform is missing")
    offset = transform.find("a:off", NS)
    extent = transform.find("a:ext", NS)
    if offset is None or extent is None:
        raise ValueError("Shape offset or extent is missing")
    return tuple(int(value) for value in (offset.get("x"), offset.get("y"), extent.get("cx"), extent.get("cy")))


def set_shape_geometry(shape: ET._Element, left: int, top: int, width: int, height: int) -> None:
    """Set shape geometry in EMUs."""
    transform = shape.find("p:spPr/a:xfrm", NS)
    if transform is None:
        raise ValueError("Shape transform is missing")
    offset = transform.find("a:off", NS)
    extent = transform.find("a:ext", NS)
    if offset is None or extent is None:
        raise ValueError("Shape offset or extent is missing")
    offset.set("x", str(left))
    offset.set("y", str(top))
    extent.set("cx", str(width))
    extent.set("cy", str(height))


def set_equation_geometry(alternate: ET._Element, left: int, top: int, width: int, height: int) -> None:
    """Set the same bounds on the editable equation and its fallback object."""
    for shape in alternate.findall("mc:Choice/p:sp", NS) + alternate.findall("mc:Fallback/p:sp", NS):
        set_shape_geometry(shape, left, top, width, height)


def set_shape_text(shape: ET._Element, text: str) -> None:
    """Replace visible text while preserving the shape's existing text style."""
    text_nodes = shape.findall(".//a:t", NS)
    if not text_nodes:
        raise ValueError("Shape has no text node")
    text_nodes[0].text = text
    if text.startswith(" ") or text.endswith(" "):
        text_nodes[0].set(XML_SPACE, "preserve")
    for node in text_nodes[1:]:
        node.text = ""


def run_properties(font_size: float, color: str, *, italic: bool, bold: bool) -> ET._Element:
    """Build DrawingML run properties for an OMML token."""
    properties = ET.Element(qn("a", "rPr"))
    properties.set("lang", "en-US")
    properties.set("altLang", "zh-CN")
    properties.set("sz", str(round(font_size * 100)))
    properties.set("b", "1" if bold else "0")
    properties.set("i", "1" if italic else "0")
    fill = ET.SubElement(properties, qn("a", "solidFill"))
    ET.SubElement(fill, qn("a", "srgbClr"), val=color.lstrip("#").upper())
    ET.SubElement(
        properties,
        qn("a", "latin"),
        typeface="Cambria Math",
        panose="02040503050406030204",
        pitchFamily="18",
        charset="0",
    )
    ET.SubElement(properties, qn("a", "ea"), typeface="Cambria Math")
    ET.SubElement(properties, qn("a", "cs"), typeface="Cambria Math")
    return properties


def math_run(text: str, font_size: float, color: str, *, italic: bool, bold: bool) -> ET._Element:
    """Build an OMML run with DrawingML formatting."""
    run = ET.Element(qn("m", "r"))
    run.append(run_properties(font_size, color, italic=italic, bold=bold))
    token = ET.SubElement(run, qn("m", "t"))
    token.text = text
    if text.startswith(" ") or text.endswith(" "):
        token.set(XML_SPACE, "preserve")
    return run


def subscript(
    base: str,
    sub: str,
    font_size: float,
    color: str,
    *,
    bold: bool,
    sub_italic: bool,
) -> ET._Element:
    """Build a structural OMML subscript node."""
    script = ET.Element(qn("m", "sSub"))
    script_properties = ET.SubElement(script, qn("m", "sSubPr"))
    control = ET.SubElement(script_properties, qn("m", "ctrlPr"))
    control.append(run_properties(font_size, color, italic=True, bold=bold))

    expression = ET.SubElement(script, qn("m", "e"))
    expression.append(math_run(base, font_size, color, italic=True, bold=bold))
    subscript_node = ET.SubElement(script, qn("m", "sub"))
    subscript_node.append(math_run(sub, font_size, color, italic=sub_italic, bold=bold))
    return script


def build_omath(tokens: tuple[tuple[str, ...], ...], font_size: float, color: str, bold: bool) -> ET._Element:
    """Build one complete editable Office Math expression."""
    equation = ET.Element(qn("m", "oMath"))
    for token in tokens:
        kind = token[0]
        if kind == "sub":
            base, sub = token[1], token[2]
            equation.append(
                subscript(
                    base,
                    sub,
                    font_size,
                    color,
                    bold=bold,
                    sub_italic=len(sub) == 1,
                )
            )
        elif kind in {"id", "op", "text"}:
            equation.append(
                math_run(
                    token[1],
                    font_size,
                    color,
                    italic=kind == "id",
                    bold=bold,
                )
            )
        else:
            raise ValueError(f"Unknown formula token kind: {kind}")
    return equation


def replace_equation_payload(
    alternate: ET._Element,
    tokens: tuple[tuple[str, ...], ...],
    font_size: float,
    color: str,
    bold: bool,
) -> None:
    """Replace an AlternateContent object's editable OMML payload."""
    math_paragraph = alternate.find("mc:Choice/p:sp/p:txBody/a:p/a14:m/m:oMathPara", NS)
    if math_paragraph is None:
        raise ValueError("Editable Office Math paragraph was not found")
    for equation in math_paragraph.findall("m:oMath", NS):
        math_paragraph.remove(equation)
    math_paragraph.append(build_omath(tokens, font_size, color, bold))

    end_properties = alternate.find("mc:Choice/p:sp/p:txBody/a:p/a:endParaRPr", NS)
    if end_properties is not None:
        end_properties.set("sz", str(round(font_size * 100)))
        end_properties.set("b", "1" if bold else "0")
        fill = end_properties.find("a:solidFill/a:srgbClr", NS)
        if fill is not None:
            fill.set("val", color.lstrip("#").upper())


def rename_equation(alternate: ET._Element, name: str, description: str) -> None:
    """Rename both editable and fallback representations."""
    for props in alternate.findall("mc:Choice/p:sp/p:nvSpPr/p:cNvPr", NS) + alternate.findall(
        "mc:Fallback/p:sp/p:nvSpPr/p:cNvPr", NS
    ):
        props.set("name", name)
        props.set("descr", description)


def formula_specs(figure: int, shapes: dict[str, ET._Element]) -> tuple[FormulaSpec, ...]:
    """Return full-expression formulas for one figure."""
    ink = "24313D"
    muted = "66717E"
    ca = "397ED1"
    refine = "F06A3B"
    white = "FFFFFF"

    def place(
        keep: str,
        anchor: str,
        tokens: tuple[tuple[str, ...], ...],
        font_size: float,
        color: str,
        left: float,
        top: float,
        width: float,
        height: float,
        bold: bool = False,
    ) -> FormulaSpec:
        if anchor not in shapes:
            raise ValueError(f"Anchor shape was not found: {anchor}")
        return FormulaSpec(keep, anchor, tokens, font_size, color, left, top, width, height, bold)

    if figure == 1:
        return (
            place("eq-figure-subtitle-regmax", "figure-subtitle-left", (("sub", "reg", "max"), ("op", "=32")), 11.0, muted, 202, 1, 82, 20),
            place("eq-ca-m-pos", "ca-formula", (("sub", "M", "pos"), ("op", "="), ("sub", "M", "in"), ("op", "∩"), ("sub", "M", "cov")), 11.0, ink, 8, 5, -16, 22),
            place("eq-ca-d-req", "ca-formula", (("sub", "D", "req"), ("op", "∕"), ("sub", "s", "k"), ("op", "≤"), ("sub", "D", "max"), ("op", "=31")), 10.5, ink, 8, 33, -16, 22),
            place("eq-ca-p3-dmax", "ca-p3", (("sub", "D", "max"), ("op", "="), ("op", "248"), ("text", " px")), 8.4, ink, 5, 22, -10, 18, True),
            place("eq-ca-p4-dmax", "ca-p4", (("sub", "D", "max"), ("op", "="), ("op", "496"), ("text", " px")), 8.4, ink, 5, 22, -10, 18, True),
            place("eq-ca-p5-dmax", "ca-p5", (("sub", "D", "max"), ("op", "="), ("op", "992"), ("text", " px")), 8.4, ink, 5, 22, -10, 18, True),
            place("eq-ref-feature-label", "ref-feature-label-placeholder", (("sub", "F", "k"),), 10.0, ink, 0, 0, 0, 18),
            place("eq-coarse-box-bc", "coarse-box", (("sub", "B", "c"), ("op", "="), ("op", "("), ("id", "x"), ("op", ","), ("id", "y"), ("op", ","), ("id", "w"), ("op", ","), ("id", "h"), ("op", ","), ("id", "θ"), ("op", ")")), 9.2, ink, 10, 34, -20, 20, True),
        )

    return (
        place("eq-offset-p-i", "offset-candidate-label", (("sub", "p", "i"), ("op", "="), ("op", "("), ("sub", "x", "f"), ("op", ","), ("sub", "y", "f"), ("op", ")")), 11.0, refine, 78, 8, 122, 22),
        place("eq-distance-left", "distance-left", (("id", "w"), ("op", "∕2+"), ("sub", "x", "f")), 9.5, refine, 4, 3, -8, -6),
        place("eq-distance-right", "distance-right", (("id", "w"), ("op", "∕2−"), ("sub", "x", "f")), 9.5, "77838F", 4, 3, -8, -6),
        place("eq-distance-top", "distance-top", (("id", "h"), ("op", "∕2−"), ("sub", "y", "f")), 9.5, "77838F", 4, 3, -8, -6),
        place("eq-distance-bottom", "distance-bottom", (("id", "h"), ("op", "∕2+"), ("sub", "y", "f")), 9.5, "77838F", 4, 3, -8, -6),
        place("eq-axis-x-label", "axis-x-label-placeholder", (("sub", "x", "f"),), 10.5, ca, 4, 2, -8, 18, True),
        place("eq-axis-y-label", "axis-y-label-placeholder", (("sub", "y", "f"),), 10.5, ca, 4, 2, -8, 18, True),
        place("eq-chart-example-d-req", "chart-example-value", (("sub", "D", "req"), ("op", "="), ("op", "360"), ("text", " px")), 10.5, ca, 8, 3, -16, -6, True),
        place("eq-chart-y-title-d-req", "chart-y-title", (("sub", "D", "req"), ("op", "∕"), ("text", "stride")), 9.0, ink, -9, 8, 18, -16),
        place("eq-threshold-label", "threshold-label", (("sub", "reg", "max"), ("op", "−1=31")), 9.0, ca, 8, 1, -16, -2, True),
        place("eq-bar-dmax-P3", "bar-dmax-P3", (("sub", "D", "max"), ("op", "="), ("op", "248"), ("text", " px")), 7.4, white, 2, -6, -4, 6, True),
        place("eq-bar-dmax-P4", "bar-dmax-P4", (("sub", "D", "max"), ("op", "="), ("op", "496"), ("text", " px")), 7.4, white, 2, -6, -4, 6, True),
        place("eq-bar-dmax-P5", "bar-dmax-P5", (("sub", "D", "max"), ("op", "="), ("op", "992"), ("text", " px")), 7.4, white, 2, -6, -4, 6, True),
        place("eq-required-d-req", "formula-required", (("sub", "D", "req"), ("op", "="), ("text", "max"), ("op", "("), ("id", "w"), ("op", "∕2+|"), ("sub", "x", "f"), ("op", "|,"), ("id", "h"), ("op", "∕2+|"), ("sub", "y", "f"), ("op", "|)")), 10.0, ink, 10, 1, -20, -2),
        place("eq-condition-d-req", "formula-condition", (("sub", "D", "req"), ("op", "∕"), ("text", "stride"), ("op", "≤"), ("sub", "reg", "max"), ("op", "−1")), 10.0, ca, 88, 1, 220, -2, True),
        place("eq-capacity-d-max", "formula-capacity", (("sub", "D", "max"), ("op", "="), ("text", "stride"), ("op", "×("), ("sub", "reg", "max"), ("op", "−1)")), 10.0, ink, 10, 1, -20, -2),
    )


def configure_anchor_aliases(figure: int, shapes: dict[str, ET._Element], equations: dict[str, ET._Element]) -> None:
    """Expose deleted source-label positions as temporary ordinary anchors."""
    aliases = (
        (("eq-ref-feature-label", "ref-feature-label-placeholder"),)
        if figure == 1
        else (("eq-axis-x-label", "axis-x-label-placeholder"), ("eq-axis-y-label", "axis-y-label-placeholder"))
    )
    for source_name, alias in aliases:
        alternate = equations[source_name]
        choice_shape = alternate.find("mc:Choice/p:sp", NS)
        if choice_shape is None:
            raise ValueError(f"Equation choice shape was not found: {source_name}")
        shapes[alias] = copy.deepcopy(choice_shape)


def remove_named_objects(root: ET._Element, names: set[str]) -> None:
    """Remove ordinary shapes or equation wrappers by object name."""
    tree = root.find(".//p:spTree", NS)
    if tree is None:
        raise ValueError("Slide shape tree was not found")
    for child in list(tree):
        name: str | None = None
        if child.tag == qn("p", "sp"):
            props = child.find("p:nvSpPr/p:cNvPr", NS)
            name = props.get("name") if props is not None else None
        elif child.tag == qn("mc", "AlternateContent"):
            props = child.find("mc:Choice/p:sp/p:nvSpPr/p:cNvPr", NS)
            name = props.get("name") if props is not None else None
        if name in names:
            tree.remove(child)


def reflow(input_path: Path, output_path: Path, figure: int) -> None:
    """Apply formula merging and save a new PPTX."""
    with zipfile.ZipFile(input_path, "r") as source:
        infos = source.infolist()
        entries = {info.filename: source.read(info.filename) for info in infos}

    slide_name = "ppt/slides/slide1.xml"
    root = ET.fromstring(entries[slide_name])
    shapes = direct_shapes(root)
    equations = equation_objects(root)
    configure_anchor_aliases(figure, shapes, equations)
    specs = formula_specs(figure, shapes)

    keep_names = {spec.keep for spec in specs}
    missing = keep_names - equations.keys()
    if missing:
        raise ValueError(f"Required seed equation objects are missing: {sorted(missing)}")

    fragment_names = {name for name in equations if name.startswith("eq-")} - keep_names
    if figure == 1:
        fragment_names |= {
            "eq-ca-mask-equals",
            "eq-ca-mask-intersection",
            "eq-ca-coverage-slash",
            "eq-ca-coverage-leq",
            "eq-ca-coverage-value",
            "eq-ca-p3-value",
            "eq-ca-p4-value",
            "eq-ca-p5-value",
            "eq-coarse-box-value",
        }
    else:
        fragment_names |= {
            "eq-offset-open",
            "eq-offset-comma",
            "eq-offset-close",
            "eq-chart-example-value-text",
            "eq-chart-y-title-stride",
            "eq-threshold-value",
            "eq-required-open",
            "eq-required-middle",
            "eq-required-close",
            "eq-condition-middle",
            "eq-condition-close",
            "eq-capacity-middle",
            "eq-capacity-close",
        }
    remove_named_objects(root, fragment_names)

    # Refresh object maps after fragment removal.
    shapes = direct_shapes(root)
    equations = equation_objects(root)
    configure_anchor_aliases(figure, shapes, equations)

    if figure == 1:
        set_shape_text(shapes["figure-subtitle-right"], "·  Fully-decoupled Refine")
        right_left, right_top, _, right_height = shape_geometry(shapes["figure-subtitle-right"])
        set_shape_geometry(shapes["figure-subtitle-right"], right_left + pt(27), right_top, pt(220), right_height)
    else:
        for name in (
            "distance-left",
            "distance-right",
            "distance-top",
            "distance-bottom",
            "chart-example-value",
            "chart-y-title",
            "threshold-label",
            "bar-dmax-P3",
            "bar-dmax-P4",
            "bar-dmax-P5",
            "formula-required",
            "formula-capacity",
        ):
            set_shape_text(shapes[name], " ")

    for spec in specs:
        alternate = equations[spec.keep]
        anchor_left, anchor_top, anchor_width, anchor_height = shape_geometry(shapes[spec.anchor])
        width = anchor_width if spec.width == 0 else (anchor_width + pt(spec.width) if spec.width < 0 else pt(spec.width))
        height = anchor_height if spec.height == 0 else (
            anchor_height + pt(spec.height) if spec.height < 0 else pt(spec.height)
        )
        left = anchor_left + pt(spec.left)
        top = anchor_top + pt(spec.top)
        set_equation_geometry(alternate, left, top, width, height)
        replace_equation_payload(alternate, spec.tokens, spec.font_size, spec.color, spec.bold)
        rename_equation(alternate, spec.keep, "Office Math full expression")

    entries[slide_name] = ET.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=output_path.stem + "_", suffix=".pptx", dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary_path, "w") as destination:
            for info in infos:
                destination.writestr(info, entries[info.filename])
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    with zipfile.ZipFile(output_path, "r") as verified:
        if verified.testzip() is not None:
            raise RuntimeError("Generated PPTX failed ZIP integrity validation")
        verified_root = ET.fromstring(verified.read(slide_name))
    final_equations = equation_objects(verified_root)
    if set(final_equations) != keep_names:
        raise RuntimeError(f"Unexpected final equation objects: {sorted(final_equations)}")
    print(f"Figure {figure}: {len(final_equations)} complete Office Math objects")
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    reflow(args.input.resolve(), args.output.resolve(), args.figure)


if __name__ == "__main__":
    main()

import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

async function loadArtifactTool() {
  try {
    return await import("@oai/artifact-tool");
  } catch (originalError) {
    for (const root of (process.env.NODE_PATH ?? "").split(path.delimiter).filter(Boolean)) {
      const candidate = path.join(root, "@oai", "artifact-tool", "dist", "artifact_tool.mjs");
      try {
        await fs.access(candidate);
        return await import(pathToFileURL(candidate).href);
      } catch {
        // Try the next bundled package root.
      }
    }
    throw originalError;
  }
}

const { Presentation, PresentationFile } = await loadArtifactTool();

const CANVAS = { width: 1800, height: 800 };
const FONT_CN = "Microsoft YaHei";
const FONT_EN = "Times New Roman";

const C = {
  bg: "#FBF8F1",
  panel: "#FFFDF8",
  white: "#FFFFFF",
  ink: "#24313D",
  muted: "#65717E",
  line: "#77838F",
  border: "#BBC3CB",
  lightBorder: "#D9DEE4",
  grid: "#DDE4EA",
  ca: "#397ED1",
  caLight: "#E3EFFC",
  refine: "#F06A3B",
  refineLight: "#FDE5DA",
  success: "#1B9C68",
  successLight: "#E3F3EC",
  grayLight: "#EEF1F4",
};

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 2) {
    args[argv[i].replace(/^--/, "")] = argv[i + 1];
  }
  return args;
}

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addShape(slide, geometry, name, position, fill, line = { style: "solid", fill: C.border, width: 1 }) {
  return slide.shapes.add({ geometry, name, position, fill, line });
}

function styleText(shape, {
  fontSize = 22,
  color = C.ink,
  bold = false,
  alignment = "center",
  verticalAlignment = "middle",
  typeface = FONT_CN,
  italic = false,
  insets = { top: 3, right: 5, bottom: 3, left: 5 },
  lineSpacing = 0.96,
} = {}) {
  shape.text.style = {
    fontSize,
    color,
    bold,
    italic,
    alignment,
    verticalAlignment,
    typeface,
    insets,
    lineSpacing,
    autoFit: "shrinkText",
  };
  return shape;
}

function addText(slide, name, text, position, options = {}) {
  const shape = addShape(slide, "textbox", name, position, "none", {
    style: "solid",
    fill: "none",
    width: 0,
  });
  shape.text = text;
  return styleText(shape, options);
}

function addBox(slide, name, text, position, {
  fill = C.white,
  stroke = C.border,
  strokeWidth = 1.2,
  radius = 16,
  shadow = undefined,
  fontSize = 21,
  color = C.ink,
  bold = false,
  alignment = "center",
  typeface = FONT_CN,
  italic = false,
  insets = { top: 4, right: 8, bottom: 4, left: 8 },
} = {}) {
  const shape = slide.shapes.add({
    geometry: "roundRect",
    name,
    position,
    fill,
    line: { style: "solid", fill: stroke, width: strokeWidth },
    borderRadius: radius,
    ...(shadow ? { shadow } : {}),
  });
  if (text !== undefined && text !== null) {
    shape.text = text;
    styleText(shape, { fontSize, color, bold, alignment, typeface, italic, insets });
  }
  return shape;
}

function addPanel(slide, name, title, position, accent) {
  addBox(slide, `${name}-panel`, null, position, {
    fill: C.panel,
    stroke: C.border,
    strokeWidth: 1.4,
    radius: 22,
  });
  addShape(slide, "rect", `${name}-accent`, {
    left: position.left,
    top: position.top,
    width: 7,
    height: position.height,
  }, accent, { style: "solid", fill: accent, width: 0 });
  addText(slide, `${name}-title`, title, {
    left: position.left + 24,
    top: position.top + 12,
    width: position.width - 48,
    height: 44,
  }, { fontSize: 29, bold: true, alignment: "left" });
}

function addAnchor(slide, name, x, y, size = 5) {
  return addShape(slide, "ellipse", name, {
    left: x - size / 2,
    top: y - size / 2,
    width: size,
    height: size,
  }, "none", { style: "solid", fill: "none", width: 0 });
}

function connect(slide, from, to, {
  color = C.line,
  width = 2,
  dashed = false,
  arrowStart = false,
  arrowEnd = true,
} = {}) {
  const connector = slide.shapes.connect(from, to, {
    kind: "straight",
    line: { style: dashed ? "dashed" : "solid", fill: color, width },
    ...(arrowStart ? { head: { type: "stealth", width: "sm", length: "sm" } } : {}),
    ...(arrowEnd ? { tail: { type: "stealth", width: "sm", length: "sm" } } : {}),
  });
  connector.bringToFront();
  return connector;
}

function addLine(slide, name, x1, y1, x2, y2, {
  color = C.line,
  width = 1.5,
  dashed = false,
} = {}) {
  return addShape(slide, "line", name, {
    left: x1,
    top: y1,
    width: x2 - x1,
    height: y2 - y1,
  }, "none", { style: dashed ? "dashed" : "solid", fill: color, width });
}

function rotatedPoint(cx, cy, x, y, angleDegrees) {
  const a = angleDegrees * Math.PI / 180;
  return {
    x: cx + x * Math.cos(a) - y * Math.sin(a),
    y: cy + x * Math.sin(a) + y * Math.cos(a),
  };
}

function addDistanceLabel(slide, name, text, position, color) {
  return addBox(slide, name, text, position, {
    fill: C.panel,
    stroke: color,
    strokeWidth: 1.1,
    radius: 10,
    fontSize: 20,
    color,
    typeface: FONT_EN,
    italic: true,
    insets: { top: 1, right: 6, bottom: 1, left: 6 },
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outputDir = path.resolve(args.outputDir ?? ".");
  const qaDir = path.resolve(args.qaDir ?? path.join(outputDir, "qa"));
  await fs.mkdir(outputDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });

  const presentation = Presentation.create({ slideSize: CANVAS });
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;

  const leftPanel = { left: 34, top: 28, width: 958, height: 660 };
  const rightPanel = { left: 1014, top: 28, width: 752, height: 660 };
  addPanel(slide, "geometry", "(a) 旋转框局部坐标与边界距离", leftPanel, C.ca);
  addPanel(slide, "levels", "(b) 特征层覆盖判定示例", rightPanel, C.refine);

  // Panel (a): connectors first, visible nodes and labels afterwards.
  const rectCenter = { x: 520, y: 386 };
  const rectAngle = -20;
  const rectWidth = 720;
  const rectHeight = 210;
  const candidate = rotatedPoint(rectCenter.x, rectCenter.y, 105, -18, rectAngle);
  const leftEdge = rotatedPoint(rectCenter.x, rectCenter.y, -rectWidth / 2, -18, rectAngle);
  const rightEdge = rotatedPoint(rectCenter.x, rectCenter.y, rectWidth / 2, -18, rectAngle);
  const topEdge = rotatedPoint(rectCenter.x, rectCenter.y, 105, -rectHeight / 2, rectAngle);
  const bottomEdge = rotatedPoint(rectCenter.x, rectCenter.y, 105, rectHeight / 2, rectAngle);

  // Draw the GT first so the distance arrows remain visible above its translucent fill.
  addShape(slide, "rect", "rotated-ground-truth", {
    left: rectCenter.x - rectWidth / 2,
    top: rectCenter.y - rectHeight / 2,
    width: rectWidth,
    height: rectHeight,
    rotation: rectAngle,
  }, C.caLight, { style: "solid", fill: C.ca, width: 4 });

  const centerAnchor = addAnchor(slide, "geometry-center-anchor", rectCenter.x, rectCenter.y);
  const candidateAnchor = addAnchor(slide, "geometry-candidate-anchor", candidate.x, candidate.y);
  const leftAnchor = addAnchor(slide, "geometry-left-edge", leftEdge.x, leftEdge.y);
  const rightAnchor = addAnchor(slide, "geometry-right-edge", rightEdge.x, rightEdge.y);
  const topAnchor = addAnchor(slide, "geometry-top-edge", topEdge.x, topEdge.y);
  const bottomAnchor = addAnchor(slide, "geometry-bottom-edge", bottomEdge.x, bottomEdge.y);
  connect(slide, centerAnchor, candidateAnchor, { color: C.muted, width: 1.8, dashed: true, arrowEnd: false });
  connect(slide, candidateAnchor, leftAnchor, { color: C.refine, width: 3.2, arrowStart: true, arrowEnd: true });
  connect(slide, candidateAnchor, rightAnchor, { color: C.line, width: 2.6, arrowStart: true, arrowEnd: true });
  connect(slide, candidateAnchor, topAnchor, { color: C.line, width: 2.6, arrowStart: true, arrowEnd: true });
  connect(slide, candidateAnchor, bottomAnchor, { color: C.line, width: 2.6, arrowStart: true, arrowEnd: true });

  const candidateLabelAnchor = addAnchor(slide, "candidate-label-anchor", 738, 166);
  connect(slide, candidateLabelAnchor, candidateAnchor, { color: C.refine, width: 1.6, arrowEnd: true });

  const axisOrigin = addAnchor(slide, "axis-origin", 178, 592);
  const axisX = addAnchor(slide, "axis-x-end", 288, 552);
  const axisY = addAnchor(slide, "axis-y-end", 141, 493);
  connect(slide, axisOrigin, axisX, { color: C.ca, width: 2.2, arrowEnd: true });
  connect(slide, axisOrigin, axisY, { color: C.ca, width: 2.2, arrowEnd: true });

  addShape(slide, "ellipse", "center-candidate", {
    left: rectCenter.x - 13,
    top: rectCenter.y - 13,
    width: 26,
    height: 26,
  }, C.success, { style: "solid", fill: C.white, width: 2.5 });
  addShape(slide, "ellipse", "offset-candidate", {
    left: candidate.x - 16,
    top: candidate.y - 16,
    width: 32,
    height: 32,
  }, C.refine, { style: "solid", fill: C.white, width: 2.5 });

  addBox(slide, "offset-candidate-label", "偏心候选  p_i = (x_f, y_f)", {
    left: 622,
    top: 112,
    width: 278,
    height: 52,
  }, {
    fill: C.refineLight,
    stroke: C.refine,
    strokeWidth: 1.3,
    radius: 14,
    fontSize: 21,
    color: C.refine,
    bold: true,
  });
  addBox(slide, "center-candidate-label", "中心候选", {
    left: 420,
    top: 474,
    width: 146,
    height: 40,
  }, {
    fill: C.successLight,
    stroke: C.success,
    strokeWidth: 1.1,
    radius: 12,
    fontSize: 20,
    color: C.success,
    bold: true,
  });

  addDistanceLabel(slide, "distance-left", "w/2 + x_f", { left: 270, top: 430, width: 150, height: 36 }, C.refine);
  addDistanceLabel(slide, "distance-right", "w/2 - x_f", { left: 750, top: 196, width: 150, height: 36 }, C.line);
  addDistanceLabel(slide, "distance-top", "h/2 - y_f", { left: 438, top: 214, width: 150, height: 36 }, C.line);
  addDistanceLabel(slide, "distance-bottom", "h/2 + y_f", { left: 670, top: 416, width: 150, height: 36 }, C.line);

  addText(slide, "axis-x-label", "x_f", { left: 286, top: 534, width: 54, height: 34 }, {
    fontSize: 20,
    color: C.ca,
    bold: true,
    typeface: FONT_EN,
    italic: true,
  });
  addText(slide, "axis-y-label", "y_f", { left: 106, top: 462, width: 54, height: 34 }, {
    fontSize: 20,
    color: C.ca,
    bold: true,
    typeface: FONT_EN,
    italic: true,
  });
  addText(slide, "axis-note", "GT 局部坐标轴", { left: 116, top: 608, width: 220, height: 32 }, {
    fontSize: 18,
    color: C.muted,
  });

  // Panel (b): a manually-authored editable chart with a dedicated title zone.
  addText(slide, "chart-subtitle", "同一目标在不同 stride 下的归一化距离需求", {
    left: 1054,
    top: 96,
    width: 430,
    height: 34,
  }, { fontSize: 18, alignment: "left", color: C.muted });
  addBox(slide, "chart-example-value", "D_req = 360 px", {
    left: 1506,
    top: 94,
    width: 214,
    height: 38,
  }, {
    fill: C.caLight,
    stroke: C.ca,
    strokeWidth: 1.1,
    radius: 12,
    fontSize: 20,
    color: C.ca,
    bold: true,
    typeface: FONT_EN,
    italic: true,
  });

  const plot = { left: 1110, top: 174, right: 1734, bottom: 616, max: 52.5 };
  const plotHeight = plot.bottom - plot.top;
  const yFor = (value) => plot.bottom - value / plot.max * plotHeight;

  for (let value = 0; value <= 50; value += 10) {
    const y = yFor(value);
    addLine(slide, `grid-${value}`, plot.left, y, plot.right, y, {
      color: value === 0 ? C.border : C.grid,
      width: value === 0 ? 1.6 : 1,
    });
    addText(slide, `grid-label-${value}`, String(value), {
      left: 1060,
      top: y - 15,
      width: 42,
      height: 30,
    }, { fontSize: 18, alignment: "right", color: C.muted, typeface: FONT_EN });
  }
  addLine(slide, "chart-y-axis", plot.left, plot.top, plot.left, plot.bottom, { color: C.border, width: 1.6 });
  addText(slide, "chart-y-title", "D_req / stride", {
    left: 1018,
    top: 302,
    width: 160,
    height: 38,
    rotation: 270,
  }, { fontSize: 22, color: C.ink, typeface: FONT_EN, italic: true });

  const thresholdY = yFor(31);
  addLine(slide, "coverage-threshold", plot.left, thresholdY, plot.right, thresholdY, {
    color: C.ca,
    width: 2.6,
    dashed: true,
  });
  addBox(slide, "threshold-label", "reg_max - 1 = 31", {
    left: 1482,
    top: thresholdY - 34,
    width: 218,
    height: 32,
  }, {
    fill: C.panel,
    stroke: C.ca,
    strokeWidth: 1,
    radius: 9,
    fontSize: 18,
    color: C.ca,
    bold: true,
    typeface: FONT_EN,
    italic: true,
    insets: { top: 0, right: 5, bottom: 0, left: 5 },
  });

  const layers = [
    { level: "P3", stride: 8, value: 45.0, dmax: 248, color: C.refine, light: C.refineLight, status: "不可达" },
    { level: "P4", stride: 16, value: 22.5, dmax: 496, color: C.success, light: C.successLight, status: "可达" },
    { level: "P5", stride: 32, value: 11.25, dmax: 992, color: C.success, light: C.successLight, status: "可达" },
  ];
  const centers = [1230, 1450, 1670];
  const barWidth = 112;

  for (let i = 0; i < layers.length; i += 1) {
    const layer = layers[i];
    const center = centers[i];
    const top = yFor(layer.value);
    addShape(slide, "rect", `bar-${layer.level}`, {
      left: center - barWidth / 2,
      top,
      width: barWidth,
      height: plot.bottom - top,
    }, layer.color, { style: "solid", fill: C.white, width: 1.1 });
    addBox(slide, `bar-value-${layer.level}`, `${layer.value.toFixed(2)}\n${layer.status}`, {
      left: center - 61,
      top: top - 62,
      width: 122,
      height: 54,
    }, {
      fill: layer.light,
      stroke: layer.color,
      strokeWidth: 1.1,
      radius: 12,
      fontSize: 19,
      color: layer.color,
      bold: true,
      insets: { top: 1, right: 4, bottom: 1, left: 4 },
    });
    addText(slide, `bar-dmax-${layer.level}`, `Dmax = ${layer.dmax} px`, {
      left: center - 55,
      top: plot.bottom - 28,
      width: 110,
      height: 22,
    }, {
      fontSize: 12.5,
      color: C.white,
      bold: true,
      typeface: FONT_EN,
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    });
    addText(slide, `bar-level-${layer.level}`, layer.level, {
      left: center - 50,
      top: plot.bottom + 8,
      width: 100,
      height: 28,
    }, { fontSize: 22, color: C.ink, bold: true, typeface: FONT_EN });
    addText(slide, `bar-stride-${layer.level}`, `stride ${layer.stride}`, {
      left: center - 62,
      top: plot.bottom + 34,
      width: 124,
      height: 26,
    }, { fontSize: 17, color: C.muted, typeface: FONT_EN });
  }

  // Shared formula band keeps symbolic definitions outside both visual panels.
  addBox(slide, "formula-band", null, { left: 34, top: 708, width: 1732, height: 66 }, {
    fill: C.panel,
    stroke: C.lightBorder,
    strokeWidth: 1.2,
    radius: 15,
  });
  addText(slide, "formula-required", "D_req = max(w/2 + |x_f|, h/2 + |y_f|)", {
    left: 66,
    top: 720,
    width: 580,
    height: 42,
  }, { fontSize: 22, color: C.ink, typeface: FONT_EN, italic: true });
  addBox(slide, "formula-condition", "可达条件：D_req / stride ≤ reg_max - 1", {
    left: 660,
    top: 720,
    width: 500,
    height: 42,
  }, {
    fill: C.caLight,
    stroke: C.ca,
    strokeWidth: 1.1,
    radius: 12,
    fontSize: 20,
    color: C.ca,
    bold: true,
  });
  addText(slide, "formula-capacity", "D_max = stride × (reg_max - 1)", {
    left: 1176,
    top: 720,
    width: 550,
    height: 42,
  }, { fontSize: 22, color: C.ink, typeface: FONT_EN, italic: true });

  const pptxPath = path.join(outputDir, "fig2_geometric_reachability.pptx");
  const previewPath = path.join(qaDir, "fig2_geometric_reachability.artifact-preview.png");
  const layoutPath = path.join(qaDir, "fig2_geometric_reachability.layout.json");
  const inspectPath = path.join(qaDir, "fig2_geometric_reachability.inspect.ndjson");

  await writeBlob(previewPath, await presentation.export({ slide, format: "png", scale: 2 }));
  await fs.writeFile(layoutPath, await (await slide.export({ format: "layout" })).text(), "utf8");
  const inspection = await presentation.inspect({ kind: "slide,textbox,shape", maxChars: 40000 });
  await fs.writeFile(inspectPath, inspection.ndjson, "utf8");
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(pptxPath);

  console.log(`PPTX: ${pptxPath}`);
  console.log(`PREVIEW: ${previewPath}`);
  console.log(`LAYOUT: ${layoutPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

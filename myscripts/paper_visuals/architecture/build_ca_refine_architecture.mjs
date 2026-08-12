import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

async function loadArtifactTool() {
  try {
    return await import("@oai/artifact-tool");
  } catch (originalError) {
    // Node ESM does not consult NODE_PATH for bare imports.  Codex's bundled
    // document runtime exposes its package root through NODE_PATH, so resolve
    // the package entry explicitly when the repo has no local node_modules.
    for (const root of (process.env.NODE_PATH ?? "").split(path.delimiter).filter(Boolean)) {
      const candidate = path.join(root, "@oai", "artifact-tool", "dist", "artifact_tool.mjs");
      try {
        await fs.access(candidate);
        return await import(pathToFileURL(candidate).href);
      } catch {
        // Keep trying the remaining NODE_PATH entries.
      }
    }
    throw originalError;
  }
}

const { Presentation, PresentationFile } = await loadArtifactTool();

const CANVAS = { width: 1920, height: 1080 };
const FONT_CN = "Microsoft YaHei";
const FONT_EN = "Times New Roman";
let LANGUAGE = "zh";

function tr(zh, en) {
  return LANGUAGE === "en" ? en : zh;
}

const C = {
  bg: "#FBF8F1",
  paper: "#FFFDF8",
  ink: "#24313D",
  muted: "#66717E",
  line: "#77838F",
  border: "#B8C0C8",
  softBorder: "#D8DDE2",
  white: "#FFFFFF",
  backbone: "#48B99A",
  backboneDark: "#22866F",
  backboneLight: "#DDF4EC",
  neck: "#3F9ED8",
  neckLight: "#E0F2FA",
  concat: "#F1A45B",
  concatLight: "#FBE8D1",
  head: "#8974B8",
  headLight: "#ECE7F7",
  box: "#45A879",
  cls: "#E85F7D",
  angle: "#E0A22C",
  refine: "#F06A3B",
  refineLight: "#FDE5DA",
  ca: "#397ED1",
  caLight: "#E3EFFC",
  gt: "#D94C4C",
  success: "#1B9C68",
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
  fontSize = 18,
  color = C.ink,
  bold = false,
  alignment = "center",
  verticalAlignment = "middle",
  typeface = FONT_CN,
  insets = { top: 4, right: 6, bottom: 4, left: 6 },
  lineSpacing = 0.95,
} = {}) {
  shape.text.style = {
    fontSize,
    color,
    bold,
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
  const shape = addShape(
    slide,
    "textbox",
    name,
    position,
    "none",
    { style: "solid", fill: "none", width: 0 },
  );
  shape.text = text;
  return styleText(shape, options);
}

function addBox(slide, name, text, position, {
  fill = C.white,
  stroke = C.border,
  strokeWidth = 1.3,
  radius = 12,
  shadow = undefined,
  fontSize = 18,
  color = C.ink,
  bold = false,
  alignment = "center",
  typeface = FONT_CN,
  insets = { top: 6, right: 8, bottom: 6, left: 8 },
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
    styleText(shape, { fontSize, color, bold, alignment, typeface, insets });
  }
  return shape;
}

function addPill(slide, name, text, position, fill, {
  stroke = fill,
  color = C.white,
  fontSize = 17,
  bold = true,
} = {}) {
  return addBox(slide, name, text, position, {
    fill,
    stroke,
    strokeWidth: 1,
    radius: 18,
    fontSize,
    color,
    bold,
    insets: { top: 1, right: 6, bottom: 1, left: 6 },
  });
}

function addAnchor(slide, name, position) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function connect(slide, from, to, {
  kind = "straight",
  fromSide = "right",
  toSide = "left",
  color = C.line,
  width = 2,
  dashed = false,
  arrow = true,
} = {}) {
  const connector = slide.shapes.connect(from, to, {
    kind,
    fromSide,
    toSide,
    line: { style: dashed ? "dashed" : "solid", fill: color, width },
    // artifact-tool maps `tail` to the destination end in exported PowerPoint connectors.
    ...(arrow ? { tail: { type: "stealth", width: "sm", length: "sm" } } : {}),
  });
  // Connectors are authored before visible nodes, but after the section backgrounds.
  // Bring them above the panels now; all subsequently-created nodes will still cover the lines.
  connector.bringToFront();
  return connector;
}

function addSection(slide, name, title, position, accent, fill = C.paper) {
  const panel = addBox(slide, `${name}-panel`, null, position, {
    fill,
    stroke: C.border,
    strokeWidth: 1.4,
    radius: 22,
    shadow: "shadow-sm",
  });
  addShape(slide, "rect", `${name}-accent`, {
    left: position.left,
    top: position.top,
    width: 8,
    height: position.height,
  }, accent, { style: "solid", fill: accent, width: 0 });
  addText(slide, `${name}-title`, title, {
    left: position.left + 22,
    top: position.top + 10,
    width: position.width - 44,
    height: 36,
  }, { fontSize: 26, bold: true, alignment: "left", color: C.ink });
  return panel;
}

function addFeatureStack(slide, name, position, color, label, detail, {
  labelTop = false,
  compact = false,
} = {}) {
  const offsets = [14, 7, 0];
  const fills = [C.grayLight, color === C.backbone ? C.backboneLight : C.neckLight, color];
  for (let i = 0; i < offsets.length; i += 1) {
    addShape(slide, "parallelogram", `${name}-layer-${i}`, {
      left: position.left + offsets[i],
      top: position.top - offsets[i],
      width: position.width,
      height: position.height,
    }, fills[i], { style: "solid", fill: color, width: i === 2 ? 1.6 : 1 });
  }
  const labelY = labelTop ? position.top - 52 : position.top + position.height + 8;
  addText(slide, `${name}-label`, label, {
    left: position.left - 18,
    top: labelY,
    width: position.width + 50,
    height: 24,
  }, { fontSize: compact ? 15 : 17, bold: true, color: C.ink });
  addText(slide, `${name}-detail`, detail, {
    left: position.left - 32,
    top: labelY + 22,
    width: position.width + 78,
    height: 38,
  }, { fontSize: compact ? 13 : 15, color: C.muted });
}

function addThreeBranchHead(slide, name, position, scaleText) {
  addBox(slide, `${name}-frame`, null, position, {
    fill: C.headLight,
    stroke: C.head,
    strokeWidth: 1.4,
    radius: 14,
  });
  addText(slide, `${name}-scale`, scaleText, {
    left: position.left + 8,
    top: position.top + 4,
    width: 48,
    height: position.height - 8,
  }, { fontSize: 17, bold: true, color: C.head });
  const x = position.left + 62;
  const y = position.top + 8;
  const w = position.width - 72;
  const h = 9;
  const gap = 8;
  const colors = [C.box, C.cls, C.angle];
  for (let i = 0; i < 3; i += 1) {
    addShape(slide, "roundRect", `${name}-branch-${i}`, {
      left: x,
      top: y + i * (h + gap),
      width: w,
      height: h,
    }, colors[i], { style: "solid", fill: colors[i], width: 0 });
  }
}

function addLegendItem(slide, name, x, y, color, text) {
  addShape(slide, "roundRect", `${name}-swatch`, { left: x, top: y + 3, width: 28, height: 12 }, color,
    { style: "solid", fill: color, width: 0 });
  addText(slide, `${name}-text`, text, { left: x + 34, top: y - 2, width: 98, height: 22 }, {
    fontSize: 14,
    alignment: "left",
    color: C.muted,
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  LANGUAGE = args.language === "en" ? "en" : "zh";
  const outputDir = path.resolve(args.outputDir ?? ".");
  const qaDir = path.resolve(args.qaDir ?? path.join(outputDir, "qa"));
  const inputImage = path.resolve(args.inputImage);
  await fs.mkdir(outputDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });

  const presentation = Presentation.create({ slideSize: CANVAS });
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;

  // Overall title: concise enough to remain outside the core architecture.
  addText(slide, "figure-title", "CA–Refine YOLO11–OBB", {
    left: 54,
    top: 20,
    width: 660,
    height: 52,
  }, { fontSize: 38, bold: true, alignment: "left", color: C.ink });
  addText(slide, "figure-subtitle", "Coverage-Aware Assignment  ·  reg_max = 32  ·  Proposal-level Geometry Refine", {
    left: 58,
    top: 68,
    width: 980,
    height: 30,
  }, { fontSize: 18, alignment: "left", color: C.muted, typeface: FONT_EN });


  // Major regions.
  addSection(slide, "input", "Input", { left: 46, top: 112, width: 248, height: 548 }, C.ink);
  addSection(slide, "backbone", "Backbone", { left: 316, top: 112, width: 548, height: 548 }, C.backbone);
  addSection(slide, "neck", "PAN–FPN Neck", { left: 886, top: 112, width: 564, height: 548 }, C.neck);
  addSection(slide, "head", "OBB Detect Head", { left: 1472, top: 112, width: 402, height: 548 }, C.head);

  // Invisible anchors are created before connectors so all edges stay behind nodes.
  const A = {};
  A.input = addAnchor(slide, "a-input", { left: 74, top: 224, width: 188, height: 188 });
  A.b1 = addAnchor(slide, "a-b1", { left: 344, top: 250, width: 76, height: 102 });
  A.b2 = addAnchor(slide, "a-b2", { left: 446, top: 258, width: 70, height: 94 });
  A.b3 = addAnchor(slide, "a-b3", { left: 548, top: 268, width: 64, height: 84 });
  A.b4 = addAnchor(slide, "a-b4", { left: 650, top: 278, width: 58, height: 74 });
  A.b5 = addAnchor(slide, "a-b5", { left: 752, top: 288, width: 52, height: 64 });
  A.n5src = addAnchor(slide, "a-n5src", { left: 912, top: 456, width: 54, height: 60 });
  A.td4 = addAnchor(slide, "a-td4", { left: 1010, top: 350, width: 64, height: 72 });
  A.p3 = addAnchor(slide, "a-p3", { left: 1120, top: 232, width: 80, height: 88 });
  A.p4 = addAnchor(slide, "a-p4", { left: 1240, top: 348, width: 66, height: 74 });
  A.p5 = addAnchor(slide, "a-p5", { left: 1360, top: 456, width: 54, height: 60 });
  A.h3 = addAnchor(slide, "a-h3", { left: 1520, top: 224, width: 248, height: 78 });
  A.h4 = addAnchor(slide, "a-h4", { left: 1520, top: 342, width: 248, height: 78 });
  A.h5 = addAnchor(slide, "a-h5", { left: 1520, top: 460, width: 248, height: 78 });

  connect(slide, A.input, A.b1, { color: C.ink, width: 2.6 });
  connect(slide, A.b1, A.b2, { color: C.backboneDark });
  connect(slide, A.b2, A.b3, { color: C.backboneDark });
  connect(slide, A.b3, A.b4, { color: C.backboneDark });
  connect(slide, A.b4, A.b5, { color: C.backboneDark });
  connect(slide, A.b5, A.n5src, { color: C.line, width: 2.4 });
  connect(slide, A.n5src, A.td4, { fromSide: "top", toSide: "bottom", color: C.neck, width: 2.2 });
  connect(slide, A.td4, A.p3, { fromSide: "top", toSide: "bottom", color: C.neck, width: 2.2 });
  connect(slide, A.p3, A.p4, { fromSide: "bottom", toSide: "top", color: C.concat, width: 2.2 });
  connect(slide, A.p4, A.p5, { fromSide: "bottom", toSide: "top", color: C.concat, width: 2.2 });
  connect(slide, A.b4, A.td4, { kind: "straight", fromSide: "right", toSide: "left", color: C.backbone, width: 1.5, dashed: true, arrow: false });
  connect(slide, A.b3, A.p3, { kind: "straight", fromSide: "right", toSide: "left", color: C.backbone, width: 1.5, dashed: true, arrow: false });
  connect(slide, A.p3, A.h3, { color: C.head, width: 2.2 });
  connect(slide, A.p4, A.h4, { color: C.head, width: 2.2 });
  connect(slide, A.p5, A.h5, { color: C.head, width: 2.2 });

  // Input image with subtle depth layers.
  addShape(slide, "roundRect", "input-shadow-2", { left: 91, top: 244, width: 178, height: 178 }, C.concatLight,
    { style: "solid", fill: C.concat, width: 1 });
  addShape(slide, "roundRect", "input-shadow-1", { left: 82, top: 234, width: 178, height: 178 }, C.backboneLight,
    { style: "solid", fill: C.backbone, width: 1 });
  const imageBytes = await fs.readFile(inputImage);
  slide.images.add({
    blob: imageBytes,
    contentType: "image/jpeg",
    alt: tr("输电线巡检图像样例", "Power-line inspection image example"),
    fit: "cover",
    geometry: "roundRect",
    borderRadius: 12,
    position: { left: 72, top: 224, width: 178, height: 178 },
  });
  addText(slide, "input-label", tr("低空巡检图像", "Low-altitude UAV image"), { left: 68, top: 432, width: 194, height: 30 }, {
    fontSize: 20,
    bold: true,
  });
  addText(slide, "input-size", "640 × 640 × 3", { left: 68, top: 463, width: 194, height: 28 }, {
    fontSize: 18,
    color: C.muted,
    typeface: FONT_EN,
  });
  addBox(slide, "input-note", tr("旋转目标保留方向信息", "Oriented boxes retain direction"), { left: 73, top: 523, width: 194, height: 66 }, {
    fill: C.grayLight,
    stroke: C.softBorder,
    fontSize: 16,
    color: C.muted,
  });

  // Backbone stage feature maps and stage operators.
  addFeatureStack(slide, "b1", { left: 344, top: 250, width: 76, height: 102 }, C.backbone, "P1 / 2", "320×320×64", { compact: true });
  addFeatureStack(slide, "b2", { left: 446, top: 258, width: 70, height: 94 }, C.backbone, "P2 / 4", "160×160×256", { compact: true });
  addFeatureStack(slide, "b3", { left: 548, top: 268, width: 64, height: 84 }, C.backbone, "P3 / 8", "80×80×512", { compact: true });
  addFeatureStack(slide, "b4", { left: 650, top: 278, width: 58, height: 74 }, C.backbone, "P4 / 16", "40×40×512", { compact: true });
  addFeatureStack(slide, "b5", { left: 752, top: 288, width: 52, height: 64 }, C.backbone, "P5 / 32", "20×20×512", { compact: true });

  const bx = [338, 440, 542, 644, 734];
  const bw = [90, 90, 90, 90, 112];
  const btxt = ["Conv", "Conv\nC3k2×2", "Conv\nC3k2×2", "Conv\nC3k2×2", "Conv · C3k2×2\nSPPF · C2PSA"];
  for (let i = 0; i < bx.length; i += 1) {
    addBox(slide, `b-op-${i}`, btxt[i], { left: bx[i], top: 455, width: bw[i], height: i === 4 ? 70 : 58 }, {
      fill: i === 4 ? C.headLight : C.backboneLight,
      stroke: i === 4 ? C.head : C.backboneDark,
      fontSize: i === 4 ? 14 : 15,
      bold: true,
      radius: 12,
    });
  }
  addText(slide, "backbone-note", tr("逐级下采样提取层次化语义特征", "Hierarchical features through progressive downsampling"), { left: 354, top: 578, width: 470, height: 32 }, {
    fontSize: 17,
    color: C.muted,
  });

  // PAN-FPN: a compact U-shaped feature path.
  addFeatureStack(slide, "n5src", { left: 912, top: 456, width: 54, height: 60 }, C.neck, "P5", "20×20", { compact: true });
  addFeatureStack(slide, "td4", { left: 1010, top: 350, width: 64, height: 72 }, C.neck, "P4ᵗᵈ", "40×40", { compact: true });
  addFeatureStack(slide, "p3", { left: 1120, top: 232, width: 80, height: 88 }, C.neck, "P3", "80×80×256", { compact: true });
  addFeatureStack(slide, "p4", { left: 1240, top: 348, width: 66, height: 74 }, C.neck, "P4", "40×40×512", { compact: true });
  addFeatureStack(slide, "p5", { left: 1360, top: 456, width: 54, height: 60 }, C.neck, "P5", "20×20×512", { compact: true });

  addPill(slide, "up-1", "↑2 + Concat", { left: 938, top: 351, width: 120, height: 30 }, C.neck, { fontSize: 14 });
  addPill(slide, "up-2", "↑2 + Concat", { left: 1047, top: 241, width: 120, height: 30 }, C.neck, { fontSize: 14 });
  addPill(slide, "down-1", "↓2 + Concat", { left: 1166, top: 349, width: 124, height: 30 }, C.concat, { color: C.ink, fontSize: 14 });
  addPill(slide, "down-2", "↓2 + Concat", { left: 1284, top: 458, width: 124, height: 30 }, C.concat, { color: C.ink, fontSize: 14 });
  addPill(slide, "neck-block", tr("各融合节点：C3k2 × 2", "Fusion block: C3k2 × 2"), { left: 1033, top: 568, width: 270, height: 34 }, C.backboneDark, { fontSize: 15 });
  addText(slide, "neck-note", tr("虚线为 Backbone 横向连接", "Dashed lines: backbone lateral links"), { left: 1072, top: 608, width: 230, height: 24 }, {
    fontSize: 14,
    color: C.muted,
  });
  addText(slide, "p5-lateral-note", "+ P5 lateral", { left: 1328, top: 534, width: 120, height: 22 }, {
    fontSize: 13,
    color: C.backboneDark,
    typeface: FONT_EN,
  });

  // Multi-scale OBB detection heads. Refine is deliberately kept outside the dense head.
  addThreeBranchHead(slide, "head-p3", { left: 1520, top: 224, width: 248, height: 78 }, "P3");
  addThreeBranchHead(slide, "head-p4", { left: 1520, top: 342, width: 248, height: 78 }, "P4");
  addThreeBranchHead(slide, "head-p5", { left: 1520, top: 460, width: 248, height: 78 }, "P5");
  addText(slide, "head-main-label", tr("每一尺度采用 Box / Cls / Angle 三分支", "Box / Cls / Angle branches at each scale"), {
    left: 1514,
    top: 552,
    width: 320,
    height: 28,
  }, { fontSize: 16, color: C.muted });
  addLegendItem(slide, "leg-box", 1518, 592, C.box, "Box / DFL");
  addLegendItem(slide, "leg-cls", 1642, 592, C.cls, "Cls");
  addLegendItem(slide, "leg-angle", 1730, 592, C.angle, "Angle");
  addPill(slide, "post-nms-tag", tr("NMS 后输出 coarse proposal", "Post-NMS coarse proposals"), { left: 1540, top: 621, width: 278, height: 26 }, C.head, { fontSize: 13 });

  // Bottom panels.
  addSection(slide, "ca-detail", tr("A  Coverage-Aware Assignment（训练阶段）", "A  Coverage-Aware Assignment (training only)"), {
    left: 46,
    top: 690,
    width: 700,
    height: 340,
  }, C.ca, C.paper);
  addSection(slide, "ref-detail", "B  Proposal-level Local Geometry Refine", {
    left: 766,
    top: 690,
    width: 1108,
    height: 340,
  }, C.refine, C.paper);

  // CA detail: grid, formula and layer reach.
  const gridX = 78;
  const gridY = 765;
  const gridW = 220;
  const gridH = 184;
  for (let i = 0; i <= 8; i += 1) {
    addShape(slide, "line", `ca-grid-v-${i}`, {
      left: gridX + (gridW / 8) * i,
      top: gridY,
      width: 0,
      height: gridH,
    }, "none", { style: "solid", fill: C.softBorder, width: 0.8 });
  }
  for (let i = 0; i <= 6; i += 1) {
    addShape(slide, "line", `ca-grid-h-${i}`, {
      left: gridX,
      top: gridY + (gridH / 6) * i,
      width: gridW,
      height: 0,
    }, "none", { style: "solid", fill: C.softBorder, width: 0.8 });
  }
  addShape(slide, "rect", "ca-long-gt", { left: 102, top: 836, width: 176, height: 18, rotation: -24 }, "#FFFFFF00",
    { style: "solid", fill: C.ink, width: 4 });
  const points = [
    [105, 902, C.gt], [139, 878, C.success], [174, 860, C.success], [213, 843, C.success], [253, 816, C.gt], [282, 888, C.gt],
  ];
  for (let i = 0; i < points.length; i += 1) {
    const [x, y, fill] = points[i];
    addShape(slide, "ellipse", `ca-point-${i}`, { left: x, top: y, width: 17, height: 17 }, fill,
      { style: "solid", fill, width: 0 });
  }
  addText(slide, "ca-grid-label", tr("细长旋转 GT 与候选点", "Elongated GT and candidates"), { left: 76, top: 950, width: 228, height: 26 }, {
    fontSize: 15,
    color: C.muted,
  });

  addText(slide, "ca-step-label", tr("覆盖可达判定", "Coverage feasibility"), { left: 325, top: 755, width: 172, height: 30 }, {
    fontSize: 19,
    bold: true,
    color: C.ca,
  });
  addBox(slide, "ca-formula", "M_pos = M_in ∩ M_cov\nD_req / s_k ≤ D_max = 31", {
    left: 318,
    top: 794,
    width: 246,
    height: 84,
  }, {
    fill: C.caLight,
    stroke: C.ca,
    strokeWidth: 1.5,
    fontSize: 17,
    bold: true,
    typeface: FONT_EN,
  });
  addBox(slide, "ca-fallback", tr("候选为空时回退到\n传统内部候选集合", "Fallback to conventional\ninside-box candidates if empty"), {
    left: 338,
    top: 894,
    width: 206,
    height: 66,
  }, {
    fill: C.grayLight,
    stroke: C.softBorder,
    fontSize: 15,
    color: C.muted,
  });

  addText(slide, "ca-route-label", tr("可覆盖层级", "Representable levels"), { left: 578, top: 755, width: 136, height: 30 }, {
    fontSize: 19,
    bold: true,
    color: C.ca,
  });
  addBox(slide, "ca-p3", "P3  ·  s=8\nD_max = 248 px", { left: 572, top: 794, width: 160, height: 58 }, {
    fill: C.neckLight, stroke: C.neck, fontSize: 14, bold: true,
  });
  addBox(slide, "ca-p4", "P4  ·  s=16\nD_max = 496 px", { left: 572, top: 865, width: 160, height: 58 }, {
    fill: C.backboneLight, stroke: C.backboneDark, fontSize: 14, bold: true,
  });
  addBox(slide, "ca-p5", "P5  ·  s=32\nD_max = 992 px", { left: 572, top: 936, width: 160, height: 58 }, {
    fill: C.concatLight, stroke: C.concat, fontSize: 14, bold: true,
  });

  // Proposal-level refine anchors and connectors first.
  const R = {};
  R.f = addAnchor(slide, "a-ref-feature", { left: 798, top: 808, width: 126, height: 96 });
  R.coarse = addAnchor(slide, "a-ref-coarse", { left: 944, top: 785, width: 204, height: 92 });
  R.roi = addAnchor(slide, "a-ref-roi", { left: 1180, top: 790, width: 178, height: 86 });
  R.fusion = addAnchor(slide, "a-ref-fusion", { left: 1390, top: 800, width: 148, height: 66 });
  R.delta = addAnchor(slide, "a-ref-delta", { left: 1572, top: 790, width: 146, height: 86 });
  R.output = addAnchor(slide, "a-ref-output", { left: 1728, top: 785, width: 122, height: 104 });
  connect(slide, R.f, R.roi, { color: C.neck, width: 2.2 });
  connect(slide, R.coarse, R.roi, { color: C.ink, width: 2.2 });
  connect(slide, R.roi, R.fusion, { color: C.neck, width: 2.2 });
  connect(slide, R.fusion, R.delta, { color: C.refine, width: 2.2 });
  connect(slide, R.delta, R.output, { color: C.refine, width: 2.2 });

  addFeatureStack(slide, "ref-feature", { left: 810, top: 814, width: 88, height: 76 }, C.neck, "F_k", "P2 / P3", { compact: true });
  addText(slide, "feature-stop-gradient-label", tr("冻结特征 · stop-grad", "Frozen features · stop-grad"), { left: 790, top: 936, width: 148, height: 22 }, {
    fontSize: 12.5,
    color: C.muted,
    typeface: FONT_CN,
  });

  addBox(slide, "coarse-box", "Coarse OBB\nB_c = (x, y, w, h, θ)", { left: 944, top: 785, width: 204, height: 92 }, {
    fill: C.backboneLight,
    stroke: C.box,
    strokeWidth: 1.6,
    fontSize: 17,
    bold: true,
    typeface: FONT_EN,
  });
  addPill(slide, "default-tag", tr("post-NMS · 全部 proposal", "post-NMS · all proposals"), { left: 949, top: 752, width: 194, height: 26 }, C.ink, { fontSize: 12.5 });

  addBox(slide, "rotated-roi", "Rotated ROI\n5 × 24", { left: 1180, top: 790, width: 178, height: 86 }, {
    fill: C.neckLight,
    stroke: C.neck,
    strokeWidth: 1.6,
    fontSize: 17,
    bold: true,
    typeface: FONT_EN,
  });
  for (let i = 1; i < 6; i += 1) {
    addShape(slide, "line", `roi-grid-v-${i}`, { left: 1195 + i * 24, top: 840, width: 0, height: 25 }, "none",
      { style: "solid", fill: C.softBorder, width: 0.7 });
  }
  addText(slide, "roi-note", tr("沿候选方向对齐采样", "Proposal-aligned sampling"), { left: 1174, top: 886, width: 190, height: 24 }, {
    fontSize: 13,
    color: C.muted,
  });

  addBox(slide, "fusion-box", tr("P2/P3 融合\nConv + MLP", "P2/P3 fusion\nConv + MLP"), { left: 1390, top: 800, width: 148, height: 66 }, {
    fill: C.caLight,
    stroke: C.ca,
    strokeWidth: 1.5,
    fontSize: 15,
    bold: true,
  });
  addBox(slide, "delta-box", tr("几何残差\nΔs, Δl", "Scale residuals\nΔs, Δl"), { left: 1572, top: 790, width: 146, height: 86 }, {
    fill: C.refineLight,
    stroke: C.refine,
    strokeWidth: 1.6,
    fontSize: 17,
    bold: true,
    typeface: FONT_EN,
  });
  addBox(slide, "ref-output", tr("Refined OBB\n\nx, y, θ 不变\ns′=s·exp(Δs)\nl′=l·exp(Δl)", "Refined OBB\n\nx, y, θ fixed\ns′=s·exp(Δs)\nl′=l·exp(Δl)"), { left: 1728, top: 785, width: 122, height: 104 }, {
    fill: C.headLight,
    stroke: C.head,
    strokeWidth: 1.7,
    fontSize: 12.5,
    bold: true,
    typeface: FONT_EN,
  });
  addPill(slide, "ref-policy", tr("中心 / 角度 / 置信度保持不变 · 不二次 NMS", "Center / angle / confidence unchanged · no second NMS"), {
    left: 1115,
    top: 934,
    width: 486,
    height: 29,
  }, C.head, { fontSize: 13.5 });
  addText(slide, "ref-loss-note", tr("训练：仅更新 Refine；目标采用分符号 tanh 平滑压缩并保留 20% 输出余量", "Training: update Refine only; sign-aware tanh target compression retains a 20% output margin"), {
    left: 1042,
    top: 983,
    width: 790,
    height: 25,
  }, { fontSize: 14, color: C.refine, alignment: "right" });

  const stem = args.stem ?? "ca_refine_architecture_redesign";
  const pngPath = path.join(outputDir, `${stem}.png`);
  const pptxPath = path.join(outputDir, `${stem}.pptx`);
  const layoutPath = path.join(qaDir, `${stem}.layout.json`);
  const inspectPath = path.join(qaDir, `${stem}.inspect.ndjson`);

  await writeBlob(pngPath, await presentation.export({ slide, format: "png", scale: 2 }));
  await fs.writeFile(layoutPath, await (await slide.export({ format: "layout" })).text(), "utf8");
  const inspection = await presentation.inspect({ kind: "slide,textbox,shape,image", maxChars: 30000 });
  await fs.writeFile(inspectPath, inspection.ndjson, "utf8");
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(pptxPath);

  console.log(`PNG: ${pngPath}`);
  console.log(`PPTX: ${pptxPath}`);
  console.log(`LAYOUT: ${layoutPath}`);
  process.exitCode = 0;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

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

function addFourBranchHead(slide, name, position, scaleText) {
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
  const gap = 5;
  const colors = [C.box, C.cls, C.angle, C.refine];
  for (let i = 0; i < 4; i += 1) {
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
  addText(slide, "figure-subtitle", "Coverage-Aware Assignment  ·  reg_max = 32  ·  Fully-decoupled Refine", {
    left: 58,
    top: 68,
    width: 980,
    height: 30,
  }, { fontSize: 18, alignment: "left", color: C.muted, typeface: FONT_EN });

  addPill(slide, "tag-main", "推理主干", { left: 1590, top: 30, width: 118, height: 34 }, C.ink, { fontSize: 16 });
  addPill(slide, "tag-train", "训练增强", { left: 1720, top: 30, width: 118, height: 34 }, C.ca, { fontSize: 16 });

  // Major regions.
  addSection(slide, "input", "Input", { left: 46, top: 112, width: 248, height: 548 }, C.ink);
  addSection(slide, "backbone", "Backbone", { left: 316, top: 112, width: 548, height: 548 }, C.backbone);
  addSection(slide, "neck", "PAN–FPN Neck", { left: 886, top: 112, width: 564, height: 548 }, C.neck);
  addSection(slide, "head", "OBBRefine Head", { left: 1472, top: 112, width: 402, height: 548 }, C.head);

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
    alt: "输电线巡检图像样例",
    fit: "cover",
    geometry: "roundRect",
    borderRadius: 12,
    position: { left: 72, top: 224, width: 178, height: 178 },
  });
  addText(slide, "input-label", "低空巡检图像", { left: 68, top: 432, width: 194, height: 30 }, {
    fontSize: 20,
    bold: true,
  });
  addText(slide, "input-size", "640 × 640 × 3", { left: 68, top: 463, width: 194, height: 28 }, {
    fontSize: 18,
    color: C.muted,
    typeface: FONT_EN,
  });
  addBox(slide, "input-note", "旋转目标保留方向信息", { left: 73, top: 523, width: 194, height: 66 }, {
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
  addText(slide, "backbone-note", "逐级下采样提取层次化语义特征", { left: 354, top: 578, width: 470, height: 32 }, {
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
  addPill(slide, "neck-block", "各融合节点：C3k2 × 2", { left: 1033, top: 568, width: 270, height: 34 }, C.backboneDark, { fontSize: 15 });
  addText(slide, "neck-note", "虚线为 Backbone 横向连接", { left: 1072, top: 608, width: 230, height: 24 }, {
    fontSize: 14,
    color: C.muted,
  });
  addText(slide, "p5-lateral-note", "+ P5 lateral", { left: 1328, top: 534, width: 120, height: 22 }, {
    fontSize: 13,
    color: C.backboneDark,
    typeface: FONT_EN,
  });

  // Multi-scale OBBRefine heads.
  addFourBranchHead(slide, "head-p3", { left: 1520, top: 224, width: 248, height: 78 }, "P3");
  addFourBranchHead(slide, "head-p4", { left: 1520, top: 342, width: 248, height: 78 }, "P4");
  addFourBranchHead(slide, "head-p5", { left: 1520, top: 460, width: 248, height: 78 }, "P5");
  addText(slide, "head-main-label", "每一尺度均采用四分支解耦预测", {
    left: 1514,
    top: 552,
    width: 320,
    height: 28,
  }, { fontSize: 16, color: C.muted });
  addLegendItem(slide, "leg-box", 1518, 592, C.box, "Box / DFL");
  addLegendItem(slide, "leg-cls", 1642, 592, C.cls, "Cls");
  addLegendItem(slide, "leg-angle", 1730, 592, C.angle, "Angle");
  addLegendItem(slide, "leg-refine", 1518, 620, C.refine, "Refine Δw, Δh");

  // Bottom panels.
  addSection(slide, "ca-detail", "A  Coverage-Aware Assignment（训练阶段）", {
    left: 46,
    top: 690,
    width: 700,
    height: 340,
  }, C.ca, C.paper);
  addSection(slide, "ref-detail", "B  Fully-decoupled Refine Head", {
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
  addText(slide, "ca-grid-label", "细长旋转 GT 与候选点", { left: 76, top: 950, width: 228, height: 26 }, {
    fontSize: 15,
    color: C.muted,
  });

  addText(slide, "ca-step-label", "覆盖可达判定", { left: 325, top: 755, width: 172, height: 30 }, {
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
  addBox(slide, "ca-fallback", "候选为空时回退到\n传统内部候选集合", {
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

  addText(slide, "ca-route-label", "可覆盖层级", { left: 578, top: 755, width: 136, height: 30 }, {
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

  // Refine detail anchors and connectors first.
  const R = {};
  R.f = addAnchor(slide, "a-ref-feature", { left: 798, top: 812, width: 126, height: 96 });
  R.box = addAnchor(slide, "a-ref-box", { left: 994, top: 752, width: 190, height: 48 });
  R.cls = addAnchor(slide, "a-ref-cls", { left: 994, top: 812, width: 190, height: 48 });
  R.ang = addAnchor(slide, "a-ref-angle", { left: 994, top: 872, width: 190, height: 48 });
  R.ref = addAnchor(slide, "a-ref-refine", { left: 994, top: 932, width: 190, height: 48 });
  R.coarse = addAnchor(slide, "a-ref-coarse", { left: 1242, top: 785, width: 222, height: 92 });
  R.delta = addAnchor(slide, "a-ref-delta", { left: 1242, top: 915, width: 222, height: 70 });
  R.gate = addAnchor(slide, "a-ref-gate", { left: 1502, top: 895, width: 166, height: 66 });
  R.output = addAnchor(slide, "a-ref-output", { left: 1700, top: 808, width: 142, height: 116 });
  connect(slide, R.f, R.box, { color: C.box, width: 2 });
  connect(slide, R.f, R.cls, { color: C.cls, width: 2 });
  connect(slide, R.f, R.ang, { color: C.angle, width: 2 });
  connect(slide, R.f, R.ref, { color: C.refine, width: 2, dashed: true });
  connect(slide, R.box, R.coarse, { color: C.box, width: 2 });
  connect(slide, R.ang, R.coarse, { color: C.angle, width: 2 });
  connect(slide, R.coarse, R.output, { color: C.ink, width: 2.4 });
  connect(slide, R.coarse, R.delta, {
    fromSide: "bottom",
    toSide: "top",
    color: C.refine,
    width: 1.7,
    dashed: true,
  });
  connect(slide, R.ref, R.delta, { color: C.refine, width: 2.2 });
  connect(slide, R.delta, R.gate, { color: C.refine, width: 2.2 });
  connect(slide, R.gate, R.output, { color: C.refine, width: 2.2 });

  addFeatureStack(slide, "ref-feature", { left: 810, top: 818, width: 88, height: 76 }, C.neck, "F_k", "P3 / P4 / P5", { compact: true });
  addPill(slide, "ref-box-branch", "Box / DFL · 4×32", { left: 994, top: 752, width: 190, height: 48 }, C.box, { fontSize: 16 });
  addPill(slide, "ref-cls-branch", "Classification · nc", { left: 994, top: 812, width: 190, height: 48 }, C.cls, { fontSize: 16 });
  addPill(slide, "ref-angle-branch", "Angle · θ", { left: 994, top: 872, width: 190, height: 48 }, C.angle, { color: C.ink, fontSize: 16 });
  addPill(slide, "ref-refine-branch", "Refine · Δw, Δh", { left: 994, top: 932, width: 190, height: 48 }, C.refine, { fontSize: 16 });
  addText(slide, "feature-stop-gradient-label", "stop-grad", { left: 924, top: 939, width: 78, height: 22 }, {
    fontSize: 12,
    color: C.refine,
    typeface: FONT_EN,
    alignment: "right",
    insets: { top: 0, right: 2, bottom: 0, left: 0 },
  });

  addBox(slide, "coarse-box", "Coarse OBB\nB_c = (x, y, w, h, θ)", { left: 1242, top: 785, width: 222, height: 92 }, {
    fill: C.backboneLight,
    stroke: C.box,
    strokeWidth: 1.6,
    fontSize: 18,
    bold: true,
    typeface: FONT_EN,
  });
  addPill(slide, "default-tag", "coarse-only（默认评估）", { left: 1248, top: 753, width: 210, height: 26 }, C.ink, { fontSize: 13 });
  addBox(slide, "delta-box", "连续宽高残差\nΔw, Δh", { left: 1242, top: 915, width: 222, height: 70 }, {
    fill: C.refineLight,
    stroke: C.refine,
    strokeWidth: 1.6,
    fontSize: 18,
    bold: true,
  });
  addText(slide, "coarse-stop-gradient-label", "stop-grad（训练）", { left: 1358, top: 881, width: 102, height: 24 }, {
    fontSize: 11.5,
    color: C.refine,
    alignment: "left",
    insets: { top: 0, right: 0, bottom: 0, left: 2 },
  });
  addBox(slide, "geometry-gate", "Geometry gate\nAR>30  or  short<16 px", { left: 1502, top: 895, width: 166, height: 66 }, {
    fill: C.concatLight,
    stroke: C.concat,
    fontSize: 13,
    bold: true,
    typeface: FONT_EN,
  });
  addBox(slide, "ref-output", "Output\n\nx, y, θ 不变\nw′ = w·exp(Δw)\nh′ = h·exp(Δh)", { left: 1700, top: 808, width: 142, height: 116 }, {
    fill: C.headLight,
    stroke: C.head,
    strokeWidth: 1.7,
    fontSize: 14,
    bold: true,
    typeface: FONT_EN,
  });
  addText(slide, "ref-loss-note", "训练：Refine 输入特征与基准粗框均停止梯度；辅助损失仅更新 Refine 分支", {
    left: 1230,
    top: 997,
    width: 620,
    height: 25,
  }, { fontSize: 14, color: C.refine, alignment: "right" });

  const pngPath = path.join(outputDir, "ca_refine_architecture_redesign.png");
  const pptxPath = path.join(outputDir, "ca_refine_architecture_redesign.pptx");
  const layoutPath = path.join(qaDir, "ca_refine_architecture_redesign.layout.json");
  const inspectPath = path.join(qaDir, "ca_refine_architecture_redesign.inspect.ndjson");

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

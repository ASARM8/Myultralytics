param(
    [string]$InputPptx = "$PSScriptRoot\..\..\..\mydocs\创新点一\paper_visuals\outputs\ca_refine_architecture_redesign.pptx",
    [string]$OutputDirectory = "$PSScriptRoot\..\..\..\mydocs\创新点一\paper_visuals\outputs"
)

$ErrorActionPreference = "Stop"
$presentation = $null
$powerPoint = $null

try {
    $resolvedInput = (Resolve-Path -LiteralPath $InputPptx).Path
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
    New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null

    $powerPoint = New-Object -ComObject PowerPoint.Application
    $presentation = $powerPoint.Presentations.Open($resolvedInput, $true, $true, $false)

    $pngPath = Join-Path $resolvedOutput "ca_refine_architecture_redesign.png"
    $pdfPath = Join-Path $resolvedOutput "ca_refine_architecture_redesign.pdf"

    # 3840x2160 provides a publication-quality raster preview.
    $presentation.Slides.Item(1).Export($pngPath, "PNG", 3840, 2160)
    # ppSaveAsPDF = 32.
    $presentation.SaveAs($pdfPath, 32)

    Write-Output "PNG: $pngPath"
    Write-Output "PDF: $pdfPath"
}
finally {
    if ($null -ne $presentation) {
        $presentation.Close()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    if ($null -ne $powerPoint) {
        $powerPoint.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

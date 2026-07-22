param(
    [Parameter(Mandatory = $true)]
    [string]$InputPptx,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
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

    $pngPath = Join-Path $resolvedOutput "fig2_geometric_reachability.png"
    $pdfPath = Join-Path $resolvedOutput "fig2_geometric_reachability.pdf"

    # The custom 1800x800 slide has an exact 2.25:1 paper-figure aspect ratio.
    $presentation.Slides.Item(1).Export($pngPath, "PNG", 3600, 1600)
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

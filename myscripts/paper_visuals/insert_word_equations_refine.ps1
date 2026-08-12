param(
    [Parameter(Mandatory = $true)]
    [string]$InputDocx
)

$ErrorActionPreference = "Stop"
$word = $null
$document = $null

function Replace-MarkerWithMath {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Marker,
        [Parameter(Mandatory = $true)][string]$Formula
    )

    foreach ($paragraph in $Document.Paragraphs) {
        $text = $paragraph.Range.Text
        $index = $text.IndexOf($Marker, [System.StringComparison]::Ordinal)
        if ($index -lt 0) {
            continue
        }

        $start = $paragraph.Range.Start + $index
        $markerRange = $Document.Range($start, $start + $Marker.Length)
        $markerRange.Text = $Formula

        $formulaRange = $Document.Range($start, $start + $Formula.Length)
        $equationRange = $Document.OMaths.Add($formulaRange)
        $equationRange.OMaths.Item(1).BuildUp()
        $equationRange.Font.Name = "Cambria Math"
        $equationRange.Font.Size = 10.5
        return
    }
    throw "Equation marker not found: $Marker"
}

try {
    $resolved = (Resolve-Path -LiteralPath $InputDocx).Path
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($resolved, $false, $false)

    # Build non-ASCII math symbols from code points so Windows PowerShell 5.1
    # never depends on the script-file encoding.
    $delta = [string][char]0x0394
    $theta = [string][char]0x03B8
    $sigma = [string][char]0x03A3
    $combiningTilde = [string][char]0x0303
    $tTilde = "t" + $combiningTilde

    $formula7 = "s_c=min(w_c,h_c),    l_c=max(w_c,h_c)"
    $formula8 = "t_s=ln(s_(gt)/s_c),    t_l=ln(l_(gt)/l_c)"
    $formula9 = "L_(ref)=(" + $sigma + "_i q_i SmoothL1(" + $delta + "_i," + $tTilde + "_i))/(" + $sigma + "_i q_i)"
    $formula10 = "s'=s_c exp(" + $delta + "s),    l'=l_c exp(" + $delta + "l),    (x',y'," + $theta + "')=(x_c,y_c," + $theta + "_c)"

    Replace-MarkerWithMath $document "[[M_BC]]" "B_c"
    Replace-MarkerWithMath $document "[[M_SC]]" "s_c"
    Replace-MarkerWithMath $document "[[M_LC]]" "l_c"
    Replace-MarkerWithMath $document "[[EQ7]]" $formula7
    Replace-MarkerWithMath $document "[[EQ8]]" $formula8
    Replace-MarkerWithMath $document "[[EQ9]]" $formula9
    Replace-MarkerWithMath $document "[[EQ10]]" $formula10

    $document.Save()
    Write-Output "Saved native Word equations: $resolved"
}
finally {
    if ($null -ne $document) {
        $document.Close()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

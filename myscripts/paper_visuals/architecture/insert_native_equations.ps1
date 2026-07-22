param(
    [Parameter(Mandatory = $true)]
    [string]$InputPptx,

    [Parameter(Mandatory = $true)]
    [string]$OutputPptx,

    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 2)]
    [int]$Figure
)

$ErrorActionPreference = 'Stop'
$powerPoint = $null
$presentation = $null
$word = $null
$document = $null

function Convert-HexToOfficeColor {
    param([Parameter(Mandatory = $true)][string]$Hex)

    $value = $Hex.TrimStart('#')
    if ($value.Length -ne 6) {
        throw "Expected a six-digit RGB color, got: $Hex"
    }
    $red = [Convert]::ToInt32($value.Substring(0, 2), 16)
    $green = [Convert]::ToInt32($value.Substring(2, 2), 16)
    $blue = [Convert]::ToInt32($value.Substring(4, 2), 16)
    return $red + ($green -shl 8) + ($blue -shl 16)
}

function Get-ShapeByName {
    param($Slide, [Parameter(Mandatory = $true)][string]$Name)

    try {
        return $Slide.Shapes.Item($Name)
    }
    catch {
        throw "Shape '$Name' was not found on slide $($Slide.SlideIndex)"
    }
}

function Remove-ExistingEquationShapes {
    param($Slide)

    for ($index = $Slide.Shapes.Count; $index -ge 1; $index--) {
        $shape = $Slide.Shapes.Item($index)
        if ($shape.Name.StartsWith('eq-', [System.StringComparison]::OrdinalIgnoreCase)) {
            $shape.Delete()
        }
    }
}

function Set-ShapeText {
    param(
        $Shape,
        [Parameter(Mandatory = $true)][string]$Text,
        [double]$FontSize,
        [string]$FontName = 'Microsoft YaHei',
        [string]$Color = '#24313D',
        [bool]$Bold = $false,
        [ValidateSet('left', 'center', 'right')][string]$Alignment = 'center',
        [double]$MarginLeft = 4,
        [double]$MarginRight = 4,
        [double]$MarginTop = 1,
        [double]$MarginBottom = 1
    )

    $Shape.TextFrame2.TextRange.Text = $Text
    $Shape.TextFrame2.MarginLeft = $MarginLeft
    $Shape.TextFrame2.MarginRight = $MarginRight
    $Shape.TextFrame2.MarginTop = $MarginTop
    $Shape.TextFrame2.MarginBottom = $MarginBottom
    $Shape.TextFrame2.VerticalAnchor = 3
    $range = $Shape.TextFrame2.TextRange
    $range.Font.Name = $FontName
    $range.Font.NameComplexScript = $FontName
    $range.Font.Size = $FontSize
    $range.Font.Bold = if ($Bold) { -1 } else { 0 }
    $range.Font.Fill.ForeColor.RGB = Convert-HexToOfficeColor $Color
    $range.ParagraphFormat.Alignment = switch ($Alignment) {
        'left' { 1 }
        'center' { 2 }
        'right' { 3 }
    }
}

function Add-PlainTextBox {
    param(
        $Slide,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Text,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$FontSize,
        [string]$FontName = 'Times New Roman',
        [string]$Color = '#24313D',
        [bool]$Bold = $false,
        [ValidateSet('left', 'center', 'right')][string]$Alignment = 'left'
    )

    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.Name = $Name
    $shape.Fill.Visible = 0
    $shape.Line.Visible = 0
    Set-ShapeText -Shape $shape -Text $Text -FontSize $FontSize -FontName $FontName -Color $Color `
        -Bold $Bold -Alignment $Alignment -MarginLeft 0 -MarginRight 0 -MarginTop 0 -MarginBottom 0
    return $shape
}

function Add-NativeEquation {
    param(
        $Slide,
        $Word,
        $Document,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$LinearText,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$FontSize,
        [string]$Color = '#24313D',
        [bool]$Bold = $false,
        [double]$Rotation = 0,
        [ValidateSet('left', 'center', 'right')][string]$Alignment = 'left'
    )

    $Document.Content.Delete()
    $Document.Activate()
    $selection = $Word.Selection
    $selection.SetRange(0, 0)
    $selection.TypeText($LinearText)
    $selection.SetRange(0, $LinearText.Length)
    $equationRange = $selection.OMaths.Add($selection.Range)
    $equationRange.OMaths.Item(1).BuildUp()
    $equation = $Document.OMaths.Item(1)
    $equation.Range.Font.Name = 'Cambria Math'
    $equation.Range.Font.Size = $FontSize
    $equation.Range.Font.Bold = if ($Bold) { -1 } else { 0 }
    $equation.Range.Font.Color = Convert-HexToOfficeColor $Color
    $paragraphAlignment = switch ($Alignment) {
        'left' { 0 }
        'center' { 1 }
        'right' { 2 }
    }
    $equation.Range.ParagraphFormat.Alignment = $paragraphAlignment
    Write-Output "Creating Office Math object: $Name = $LinearText"
    $shapeRange = $null
    for ($attempt = 1; $attempt -le 12; $attempt++) {
        try {
            $equation.Range.Copy()
            Start-Sleep -Milliseconds 300
            try { $script:powerPoint.Activate() } catch {}
            try { $Slide.Select() } catch {}
            $shapeRange = $Slide.Shapes.Paste()
            break
        }
        catch {
            if ($attempt -eq 12) { throw }
            Start-Sleep -Milliseconds 300
        }
    }
    $shape = $shapeRange.Item(1)
    $naturalHeight = [double]$shape.Height
    $shape.Name = $Name
    $shape.AlternativeText = "Office Math: $LinearText"
    $shape.LockAspectRatio = 0
    # Word places copied OMath inside a wide paragraph-sized textbox. Disable wrapping before
    # narrowing that textbox, otherwise PowerPoint breaks even short equations across lines.
    $shape.TextFrame2.WordWrap = 0
    $shape.TextFrame2.MarginLeft = 0
    $shape.TextFrame2.MarginRight = 0
    $shape.TextFrame2.MarginTop = 0
    $shape.TextFrame2.MarginBottom = 0
    $shape.TextFrame2.VerticalAnchor = 3
    $shape.Left = $Left
    $shape.Width = $Width
    $finalHeight = [Math]::Min([double]$Height, $naturalHeight)
    $shape.Height = $finalHeight
    $shape.Top = $Top + (($Height - $finalHeight) / 2)
    $shape.Rotation = $Rotation
    $shape.Fill.Visible = 0
    $shape.Line.Visible = 0
    try {
        $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = Convert-HexToOfficeColor $Color
        $shape.TextFrame2.TextRange.Font.Bold = if ($Bold) { -1 } else { 0 }
        $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = switch ($Alignment) {
            'left' { 1 }
            'center' { 2 }
            'right' { 3 }
        }
    }
    catch {}
    $shape.ZOrder(0)
    return $shape
}

function Add-NativeSubscript {
    param(
        $Slide,
        $Word,
        $Document,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Base,
        [Parameter(Mandatory = $true)][string]$Subscript,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$FontSize,
        [string]$Color = '#24313D',
        [bool]$Bold = $false,
        [double]$Rotation = 0
    )

    $Document.Content.Delete()
    $Document.Activate()
    $selection = $Word.Selection
    $selection.SetRange(0, 0)
    $selection.TypeText('x')
    $selection.SetRange(0, 1)
    $equationRange = $selection.OMaths.Add($selection.Range)
    $equation = $Document.OMaths.Item(1)
    # wdOMathFunctionScrSub = 17. Building the script object explicitly is reliable for
    # multi-character subscripts such as req/max/pos/cov, independent of Word input mode.
    $scriptFunction = $equation.Functions.Add($equation.Range, 17)
    $scriptFunction.ScrSub.E.Range.Text = $Base
    $scriptFunction.ScrSub.Sub.Range.Text = $Subscript
    $equation.Range.Font.Name = 'Cambria Math'
    $equation.Range.Font.Size = $FontSize
    $equation.Range.Font.Bold = if ($Bold) { -1 } else { 0 }
    $equation.Range.Font.Color = Convert-HexToOfficeColor $Color
    $equation.Range.ParagraphFormat.Alignment = 1

    Write-Output "Creating structural Office Math subscript: $Name = $Base / $Subscript"
    $shapeRange = $null
    for ($attempt = 1; $attempt -le 12; $attempt++) {
        try {
            $equation.Range.Copy()
            Start-Sleep -Milliseconds 300
            try { $script:powerPoint.Activate() } catch {}
            try { $Slide.Select() } catch {}
            $shapeRange = $Slide.Shapes.Paste()
            break
        }
        catch {
            if ($attempt -eq 12) { throw }
            Start-Sleep -Milliseconds 300
        }
    }

    $shape = $shapeRange.Item(1)
    $naturalHeight = [double]$shape.Height
    $shape.Name = $Name
    $shape.AlternativeText = "Office Math structured subscript: $Base / $Subscript"
    $shape.LockAspectRatio = 0
    $shape.TextFrame2.WordWrap = 0
    $shape.TextFrame2.MarginLeft = 0
    $shape.TextFrame2.MarginRight = 0
    $shape.TextFrame2.MarginTop = 0
    $shape.TextFrame2.MarginBottom = 0
    $shape.Left = $Left
    $shape.Width = $Width
    $finalHeight = [Math]::Min([double]$Height, $naturalHeight)
    $shape.Height = $finalHeight
    $shape.Top = $Top + (($Height - $finalHeight) / 2)
    $shape.Rotation = $Rotation
    $shape.Fill.Visible = 0
    $shape.Line.Visible = 0
    try {
        $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = Convert-HexToOfficeColor $Color
        $shape.TextFrame2.TextRange.Font.Bold = if ($Bold) { -1 } else { 0 }
    }
    catch {}
    $shape.ZOrder(0)
    return $shape
}

function Replace-PlainShapeWithEquation {
    param(
        $Slide,
        $Word,
        $Document,
        [Parameter(Mandatory = $true)][string]$ShapeName,
        [Parameter(Mandatory = $true)][string]$LinearText,
        [double]$FontSize,
        [string]$Color = '#24313D',
        [bool]$Bold = $false,
        [Nullable[double]]$Top = $null,
        [Nullable[double]]$Height = $null
    )

    $source = Get-ShapeByName -Slide $Slide -Name $ShapeName
    $leftValue = [double]$source.Left
    $topValue = if ($null -ne $Top) { [double]$Top } else { [double]$source.Top }
    $widthValue = [double]$source.Width
    $heightValue = if ($null -ne $Height) { [double]$Height } else { [double]$source.Height }
    $rotationValue = [double]$source.Rotation
    $source.Delete()
    Add-NativeEquation -Slide $Slide -Word $Word -Document $Document -Name "eq-$ShapeName" `
        -LinearText $LinearText -Left $leftValue -Top $topValue -Width $widthValue -Height $heightValue `
        -FontSize $FontSize -Color $Color -Bold $Bold -Rotation $rotationValue | Out-Null
}

function Apply-Figure1Equations {
    param($Slide, $Word, $Document)

    $ink = '#24313D'
    $muted = '#66717E'

    $subtitle = Get-ShapeByName -Slide $Slide -Name 'figure-subtitle'
    $subtitleLeft = [double]$subtitle.Left
    $subtitleTop = [double]$subtitle.Top
    $subtitle.Delete()
    Add-PlainTextBox -Slide $Slide -Name 'figure-subtitle-left' -Text 'Coverage-Aware Assignment  ·' `
        -Left $subtitleLeft -Top $subtitleTop -Width 205 -Height 22.5 -FontSize 13.5 -Color $muted | Out-Null
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-figure-subtitle-regmax' `
        -Base 'reg' -Subscript 'max' -Left ($subtitleLeft + 202) -Top ($subtitleTop + 1) -Width 48 -Height 18 `
        -FontSize 5.2 -Color $muted | Out-Null
    Add-PlainTextBox -Slide $Slide -Name 'figure-subtitle-right' -Text '·  Fully-decoupled Refine' `
        -Left ($subtitleLeft + 264) -Top $subtitleTop -Width 220 -Height 22.5 -FontSize 13.5 -Color $muted | Out-Null

    $formulaBox = Get-ShapeByName -Slide $Slide -Name 'ca-formula'
    Set-ShapeText -Shape $formulaBox -Text ' ' -FontSize 1 -FontName 'Times New Roman' -Color $ink
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-ca-m-pos' `
        -Base 'M' -Subscript 'pos' -Left ($formulaBox.Left + 8) -Top ($formulaBox.Top + 5) `
        -Width 30 -Height 22 -FontSize 4.15 -Color $ink | Out-Null
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-ca-d-req' `
        -Base 'D' -Subscript 'req' -Left ($formulaBox.Left + 8) -Top ($formulaBox.Top + 33) `
        -Width 30 -Height 22 -FontSize 4.05 -Color $ink | Out-Null

    foreach ($layer in @(
        @{ Name = 'ca-p3'; Header = 'P3  ·  s=8' },
        @{ Name = 'ca-p4'; Header = 'P4  ·  s=16' },
        @{ Name = 'ca-p5'; Header = 'P5  ·  s=32' }
    )) {
        $box = Get-ShapeByName -Slide $Slide -Name $layer.Name
        Set-ShapeText -Shape $box -Text $layer.Header -FontSize 10.5 -FontName 'Microsoft YaHei' -Color $ink `
            -Bold $true -Alignment 'center' -MarginTop 1 -MarginBottom 19
        Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name "eq-$($layer.Name)-dmax" `
            -Base 'D' -Subscript 'max' -Left ($box.Left + 5) -Top ($box.Top + 22) -Width 30 `
            -Height 18 -FontSize 3.05 -Color $ink -Bold $true | Out-Null
    }

    $feature = Get-ShapeByName -Slide $Slide -Name 'ref-feature-label'
    $featureLeft = [double]$feature.Left
    $featureTop = [double]$feature.Top
    $feature.Delete()
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-ref-feature-label' `
        -Base 'F' -Subscript 'k' -Left ($featureLeft + 28) -Top $featureTop -Width 44 -Height 18 `
        -FontSize 4.2 -Color $ink | Out-Null

    $coarse = Get-ShapeByName -Slide $Slide -Name 'coarse-box'
    Set-ShapeText -Shape $coarse -Text 'Coarse OBB' -FontSize 13.5 -FontName 'Times New Roman' -Color $ink `
        -Bold $true -Alignment 'center' -MarginTop 3 -MarginBottom 32
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-coarse-box-bc' `
        -Base 'B' -Subscript 'c' -Left ($coarse.Left + 10) -Top ($coarse.Top + 34) `
        -Width 28 -Height 20 -FontSize 3.65 -Color $ink -Bold $true | Out-Null

}

function Apply-Figure2Equations {
    param($Slide, $Word, $Document)

    $ink = '#24313D'
    $line = '#77838F'
    $ca = '#397ED1'
    $refine = '#F06A3B'
    $white = '#FFFFFF'

    $candidate = Get-ShapeByName -Slide $Slide -Name 'offset-candidate-label'
    Set-ShapeText -Shape $candidate -Text '偏心候选' -FontSize 12 -FontName 'Microsoft YaHei' -Color $refine `
        -Bold $true -Alignment 'left' -MarginLeft 10 -MarginRight 145
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-offset-p-i' `
        -Base 'p' -Subscript 'i' -Left ($candidate.Left + 78) -Top ($candidate.Top + 8) `
        -Width 23 -Height 22 -FontSize 4.65 -Color $refine | Out-Null

    foreach ($distance in @(
        @{ Name = 'distance-left'; Base = 'x'; Color = $refine },
        @{ Name = 'distance-right'; Base = 'x'; Color = $line },
        @{ Name = 'distance-top'; Base = 'y'; Color = $line },
        @{ Name = 'distance-bottom'; Base = 'y'; Color = $line }
    )) {
        $box = Get-ShapeByName -Slide $Slide -Name $distance.Name
        Set-ShapeText -Shape $box -Text ' ' -FontSize 1 -FontName 'Cambria Math' -Color $distance.Color
        Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name "eq-$($distance.Name)" `
            -Base $distance.Base -Subscript 'f' -Left ($box.Left + 4) -Top ($box.Top + 3) -Width 22 `
            -Height ($box.Height - 6) -FontSize 3.9 -Color $distance.Color | Out-Null
    }

    $axisX = Get-ShapeByName -Slide $Slide -Name 'axis-x-label'
    $axisXLeft = [double]$axisX.Left
    $axisXTop = [double]$axisX.Top
    $axisX.Delete()
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-axis-x-label' `
        -Base 'x' -Subscript 'f' -Left ($axisXLeft + 4) -Top ($axisXTop + 2) -Width 32 -Height 18 `
        -FontSize 4.7 -Color $ca -Bold $true | Out-Null
    $axisY = Get-ShapeByName -Slide $Slide -Name 'axis-y-label'
    $axisYLeft = [double]$axisY.Left
    $axisYTop = [double]$axisY.Top
    $axisY.Delete()
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-axis-y-label' `
        -Base 'y' -Subscript 'f' -Left ($axisYLeft + 4) -Top ($axisYTop + 2) -Width 32 -Height 18 `
        -FontSize 4.7 -Color $ca -Bold $true | Out-Null

    $example = Get-ShapeByName -Slide $Slide -Name 'chart-example-value'
    Set-ShapeText -Shape $example -Text ' ' -FontSize 1 -FontName 'Times New Roman' -Color $ca
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-chart-example-d-req' `
        -Base 'D' -Subscript 'req' -Left ($example.Left + 8) -Top ($example.Top + 3) `
        -Width 34 -Height ($example.Height - 6) -FontSize 4.45 -Color $ca -Bold $true | Out-Null

    $yTitle = Get-ShapeByName -Slide $Slide -Name 'chart-y-title'
    Set-ShapeText -Shape $yTitle -Text ' ' -FontSize 1 -FontName 'Cambria Math' -Color $ink
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-chart-y-title-d-req' `
        -Base 'D' -Subscript 'req' -Left 751 -Top 237 -Width 42 -Height 22 `
        -FontSize 3.45 -Color $ink | Out-Null

    $threshold = Get-ShapeByName -Slide $Slide -Name 'threshold-label'
    Set-ShapeText -Shape $threshold -Text ' ' -FontSize 1 -FontName 'Times New Roman' -Color $ca
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-threshold-label' `
        -Base 'reg' -Subscript 'max' -Left ($threshold.Left + 8) -Top ($threshold.Top + 1) `
        -Width 42 -Height ($threshold.Height - 2) -FontSize 3.25 -Color $ca -Bold $true | Out-Null

    foreach ($bar in @(
        @{ Name = 'bar-dmax-P3' },
        @{ Name = 'bar-dmax-P4' },
        @{ Name = 'bar-dmax-P5' }
    )) {
        $barText = Get-ShapeByName -Slide $Slide -Name $bar.Name
        Set-ShapeText -Shape $barText -Text ' ' -FontSize 1 -FontName 'Times New Roman' -Color $white
        Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name "eq-$($bar.Name)" `
            -Base 'D' -Subscript 'max' -Left ($barText.Left + 2) -Top ($barText.Top - 6) `
            -Width 25 -Height ($barText.Height + 6) -FontSize 2.45 -Color $white -Bold $true | Out-Null
    }

    $required = Get-ShapeByName -Slide $Slide -Name 'formula-required'
    Set-ShapeText -Shape $required -Text ' ' -FontSize 1 -FontName 'Cambria Math' -Color $ink
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-required-d-req' `
        -Base 'D' -Subscript 'req' -Left ($required.Left + 10) -Top ($required.Top + 1) `
        -Width 30 -Height ($required.Height - 2) -FontSize 3.35 -Color $ink | Out-Null

    $condition = Get-ShapeByName -Slide $Slide -Name 'formula-condition'
    Set-ShapeText -Shape $condition -Text '可达条件：' -FontSize 15 -FontName 'Microsoft YaHei' -Color $ca `
        -Bold $true -Alignment 'left' -MarginLeft 16 -MarginRight 310
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-condition-d-req' `
        -Base 'D' -Subscript 'req' -Left ($condition.Left + 96) -Top ($condition.Top + 1) `
        -Width 30 -Height ($condition.Height - 2) -FontSize 3.45 -Color $ca -Bold $true | Out-Null

    $capacity = Get-ShapeByName -Slide $Slide -Name 'formula-capacity'
    Set-ShapeText -Shape $capacity -Text ' ' -FontSize 1 -FontName 'Cambria Math' -Color $ink
    Add-NativeSubscript -Slide $Slide -Word $Word -Document $Document -Name 'eq-capacity-d-max' `
        -Base 'D' -Subscript 'max' -Left ($capacity.Left + 10) -Top ($capacity.Top + 1) `
        -Width 31 -Height ($capacity.Height - 2) -FontSize 3.35 -Color $ink | Out-Null
}

try {
    $resolvedInput = (Resolve-Path -LiteralPath $InputPptx).Path
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputPptx)
    $outputDirectory = Split-Path -Parent $resolvedOutput
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Add()

    $powerPoint = New-Object -ComObject PowerPoint.Application
    $powerPoint.Visible = -1
    $presentation = $powerPoint.Presentations.Open($resolvedInput, -1, 0, -1)
    if ($presentation.Slides.Count -ne 1) {
        throw "Expected a one-slide figure deck, found $($presentation.Slides.Count) slides"
    }
    $slide = $presentation.Slides.Item(1)
    Remove-ExistingEquationShapes -Slide $slide

    if ($Figure -eq 1) {
        Apply-Figure1Equations -Slide $slide -Word $word -Document $document
    }
    else {
        Apply-Figure2Equations -Slide $slide -Word $word -Document $document
    }

    $presentation.SaveAs($resolvedOutput, 24)
    Write-Output "Saved native-equation PPTX: $resolvedOutput"
}
finally {
    if ($presentation) {
        try { $presentation.Close() } catch {}
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    if ($powerPoint) {
        try { $powerPoint.Quit() } catch {}
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint)
    }
    if ($document) {
        try { $document.Close([ref]0) } catch {}
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($word) {
        try { $word.Quit() } catch {}
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

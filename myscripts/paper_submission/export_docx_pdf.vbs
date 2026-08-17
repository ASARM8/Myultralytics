Option Explicit

Dim args, word, i, docxPath, pdfPath, doc
Set args = WScript.Arguments

If args.Count = 0 Or (args.Count Mod 2) <> 0 Then
    WScript.Echo "Usage: cscript //nologo export_docx_pdf.vbs INPUT.docx OUTPUT.pdf [INPUT.docx OUTPUT.pdf ...]"
    WScript.Quit 2
End If

Set word = CreateObject("Word.Application")
word.Visible = False
word.DisplayAlerts = 0

On Error Resume Next
For i = 0 To args.Count - 1 Step 2
    docxPath = args(i)
    pdfPath = args(i + 1)
    Err.Clear
    Set doc = word.Documents.Open(docxPath, False, True)
    If Err.Number <> 0 Then
        WScript.Echo "OPEN FAILED: " & Err.Description
        word.Quit
        WScript.Quit 3
    End If
    doc.Fields.Update
    doc.ExportAsFixedFormat pdfPath, 17
    If Err.Number <> 0 Then
        WScript.Echo "EXPORT FAILED: " & Err.Description
        doc.Close False
        word.Quit
        WScript.Quit 4
    End If
    doc.Close False
    WScript.Echo pdfPath
Next

word.Quit
Set word = Nothing
On Error GoTo 0

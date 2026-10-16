param([string]$InputArchive,[string]$RootMetadata,[Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if(-not $InputArchive){$InputArchive=Join-Path $repo 'results/binary_v1/binary_masks.npz'}
if(-not $RootMetadata){$RootMetadata=Join-Path $repo 'results/binary_v1/roots.csv'}
$InputArchive=(Resolve-Path $InputArchive).Path;$RootMetadata=(Resolve-Path $RootMetadata).Path
$manifestPath=Join-Path $repo 'results/binary_v1/manifest.json'
$manifest=Get-Content -Raw $manifestPath|ConvertFrom-Json
foreach($pair in @(@('binary_masks.npz',$InputArchive),@('roots.csv',$RootMetadata))){
 $pin=$manifest.files|Where-Object path -eq $pair[0]
 if(@($pin).Count -ne 1 -or (Get-Item $pair[1]).Length -ne $pin.bytes -or (Get-FileHash $pair[1] -Algorithm SHA256).Hash.ToLower() -ne $pin.sha256){throw "Release input differs: $($pair[0])"}
}
if(Test-Path -LiteralPath $Output){throw 'Output exists; use a new directory.'}
$timer=[Diagnostics.Stopwatch]::StartNew()
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -Path (Join-Path $PSScriptRoot 'SemanticSensitivity.cs')
$models=[SemanticSensitivity]::Strings($InputArchive,'models.npy');$rootIds=[SemanticSensitivity]::Strings($InputArchive,'root_ids.npy');$conditions=[SemanticSensitivity]::Strings($InputArchive,'conditions.npy');$flags=[SemanticSensitivity]::ReadNpy($InputArchive,'flags.npy')
if(($models -join '|') -ne 'blip2|llava|qwen25|qwen3vl' -or $flags.Descriptor -ne '|u1' -or ($flags.Shape -join ',') -ne '4,1500,441' -or $flags.Data.Length -ne 2646000){throw 'Release array schema differs'}
$metadata=@{};Import-Csv $RootMetadata|ForEach-Object {if($metadata.ContainsKey($_.root)){throw 'Duplicate root'};$metadata[$_.root]=$_}
[string[]]$images=foreach($id in $rootIds){if(-not $metadata.ContainsKey($id)){throw 'Root missing identity'};$metadata[$id].image_group}
if($metadata.Count -ne 1500 -or @($images|Select-Object -Unique).Count -ne 1065){throw 'Root/image cardinality differs'}
$imageOps=@('gaussian_blur','gaussian_noise','jpeg','brightness','contrast');$textOps=@('typo','keyboard','ocr','whitespace_punc','casing')
$want=[Collections.Generic.List[string]]::new();$want.Add('clean')
foreach($u in $imageOps){foreach($i in 1..4){$want.Add("image__${u}__${i}")}}
foreach($v in $textOps){foreach($j in 1..4){$want.Add("text__${v}__${j}")}}
$ia=[Collections.Generic.List[int]]::new();$ta=[Collections.Generic.List[int]]::new();$diagonal=[Collections.Generic.List[bool]]::new();$corner=[Collections.Generic.List[bool]]::new()
for($u=0;$u -lt 5;$u++){for($v=0;$v -lt 5;$v++){foreach($i in 1..4){foreach($j in 1..4){
 $want.Add(('mixed__{0}_{1}__{2}_{3}' -f $imageOps[$u],$textOps[$v],$i,$j));$ia.Add(4*$u+$i);$ta.Add(20+4*$v+$j);$diagonal.Add($i -eq $j);$corner.Add($i -eq 4 -and $j -eq 4)
}}}}
if(($want -join '|') -ne ($conditions -join '|')){throw 'Condition order differs'}
$analysis=[SemanticSensitivity]::Analyze($flags.Data,$models,$rootIds,$images,40,$ia.ToArray(),$ta.ToArray(),$diagonal.ToArray(),$corner.ToArray())
# Verify primary counts and both-axis unconditional accuracy before exporting.
$checks=0;$old=Import-Csv (Join-Path $repo 'results/binary_v1/model_summary.csv');$axes=Import-Csv (Join-Path $repo 'results/binary_v1/accuracy_axes.csv')
foreach($s in $old){$r=$analysis['summaries']|Where-Object {$_.axis -eq 'task_pass' -and $_.model -eq $s.model};foreach($p in @(@('qualified_roots','strict41_roots'),@('affected_roots','affected_strict41_roots'),@('failure_events','strict41_failure_events'),@('all_suite_pass','all_suite_pass_roots'),@('cell_eligible','all_exact_gap_eligible'),@('cell_gaps','all_exact_gap_events'))){if($r[$p[0]] -ne [int]$s.($p[1])){throw 'Primary source summary mismatch'};$checks++}}
foreach($s in @($axes|Where-Object {$_.axis -in @('task_pass','semantic_correct') -and $_.subset -in @('clean','single','joint')})){$r=$analysis['summaries']|Where-Object {$_.axis -eq $s.axis -and $_.model -eq $s.model};if($r["$($s.subset)_correct"] -ne [int]$s.correct){throw 'Unconditional source summary mismatch'};$checks++}
[void](New-Item -ItemType Directory -Path $Output)
foreach($name in @('summaries','fixed_primary_cohorts','replacements','root_records')){
 @($analysis[$name]|ForEach-Object {$record=[ordered]@{};foreach($key in $_.Keys){$record[$key]=$_.get_Item($key)};[pscustomobject]$record})|Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $Output "$name.csv")
}
$result=[ordered]@{status='completed';rows=2646000;roots=1500;images=1065;existing_source_checks=$checks;seconds=$timer.Elapsed.TotalSeconds;model_calls=0;judge_calls=0;human_audit=$false;intervals_computed=$false;environment=[ordered]@{powershell=$PSVersionTable.PSVersion.ToString();dotnet=[Environment]::Version.ToString()};inputs=[ordered]@{archive_sha256=(Get-FileHash $InputArchive -Algorithm SHA256).Hash.ToLower();roots_sha256=(Get-FileHash $RootMetadata -Algorithm SHA256).Hash.ToLower()};source_sha256=(Get-FileHash (Join-Path $PSScriptRoot 'SemanticSensitivity.cs') -Algorithm SHA256).Hash.ToLower()}
$result|ConvertTo-Json -Depth 6|Set-Content -Encoding UTF8 (Join-Path $Output 'result.json')
$result|ConvertTo-Json -Depth 6

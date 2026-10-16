$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$source=Join-Path $PSScriptRoot 'SemanticSensitivity.cs'
if(-not (Test-Path -LiteralPath $source)){throw 'RED: SemanticSensitivity.Analyze is not implemented; fixture expectations are frozen below.'}
Add-Type -Path $source
function Equal($name,$actual,$expected){if($actual -ne $expected){throw "$name actual=$actual expected=$expected"}}
# Two models, three roots; clean + two singles + a 2x2 joint grid.
# A has one semantic off-diagonal error and a separate format-only diagonal error.
# B has a format-only clean error. C has a format-only maximum-corner error.
[byte[]]$flags=New-Object byte[] 42
for($i=0;$i -lt $flags.Length;$i++){$flags[$i]=15}
$flags[3]=9;$flags[4]=10;$flags[7]=9;$flags[20]=9
$flags[31]=10 # target B: first joint error; clean/singles succeed
$r=[SemanticSensitivity]::Analyze($flags,@('reference','target'),@('A','B','C'),@('imageA','imageB','imageC'),2,@(1,1,1,1),@(2,2,2,2),@($true,$false,$false,$true),@($false,$false,$false,$true))
$task=$r['summaries']|Where-Object {$_.axis -eq 'task_pass' -and $_.model -eq 'reference'}
$sem=$r['summaries']|Where-Object {$_.axis -eq 'semantic_correct' -and $_.model -eq 'reference'}
Equal 'primary qualification excludes format-only clean error' $task.qualified_roots 2
Equal 'semantic qualification includes B' $sem.qualified_roots 3
Equal 'primary affected A and C' $task.affected_roots 2
Equal 'semantic affected A only' $sem.affected_roots 1
Equal 'semantic off-diagonal event' $sem.failure_events 1
Equal 'semantic diagonal misses A' $sem.diagonal_missed 1
Equal 'semantic corner misses A' $sem.corner_missed 1
Equal 'primary diagonal finds A and C' $task.diagonal_found 2
Equal 'primary corner finds C' $task.corner_found 1
$fixed=$r['fixed_primary_cohorts']|Where-Object model -eq reference
Equal 'fixed primary population stays two' $fixed.qualified_roots 2
Equal 'fixed primary semantic affected A' $fixed.semantic_affected_roots 1
Equal 'format-only affected C disappears' $fixed.format_only_affected_roots 1
Equal 'cell eligibility does not silently require clean' $task.cell_eligible 12
Equal 'joint task gap vs semantic gap' $task.cell_gaps 3
Equal 'semantic joint gaps' $sem.cell_gaps 1
$repl=$r['replacements']|Where-Object {$_.axis -eq 'semantic_correct' -and $_.reference -eq 'reference' -and $_.cohort -eq 'reference_full_suite'}
Equal 'semantic old full-suite cohort B,C' $repl.denominator 2
Equal 'fixed reference catches new joint error B' $repl.joint_regression 1
Equal 'semantic old success C preserved' $repl.preserved 1
$common=$r['replacements']|Where-Object {$_.axis -eq 'semantic_correct' -and $_.reference -eq 'reference' -and $_.cohort -eq 'common_strict41'}
Equal 'common semantic roots' $common.denominator 3
Equal 'common stable roots exclude repaired root' $common.preserved 1
Equal 'common newly failed roots exclude persistent failures' $common.joint_regression 1
Equal 'semantic joint event repaired A' $common.repaired_conditions 1
Equal 'semantic joint event newly wrong B' $common.new_failed_conditions 1
# A malformed primary bit must never pass as semantic sensitivity input.
$bad=[byte[]]$flags.Clone();$bad[0]=12
$rejected=$false
try{[void][SemanticSensitivity]::Analyze($bad,@('reference','target'),@('A','B','C'),@('imageA','imageB','imageC'),2,@(1,1,1,1),@(2,2,2,2),@($true,$false,$false,$true),@($false,$false,$false,$true))}catch{$rejected=$true}
Equal 'reject task not semantic AND format' $rejected $true
[pscustomobject]@{status='PASS';assertions=24;model_calls=0;fixtures='axis qualification, off-diagonal misses, fixed primary cohort, exact anchors, reference replacement, malformed conjunction'}|ConvertTo-Json

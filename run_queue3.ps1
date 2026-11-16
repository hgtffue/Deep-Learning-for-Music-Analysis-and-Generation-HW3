# Seed-1 replicates of the key settings, then MQAScore on everything (one heavy job at a time).
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = 1
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$py = "$env:USERPROFILE\miniconda3\envs\deepmir_hw3\python.exe"
Set-Location $PSScriptRoot

$runs = @(
    @("cfg7_seed1",       @("--scale", "7", "--seed", "1")),
    @("compose7_3_seed1", @("--scale", "7", "--concepts", "concepts.csv", "--concept_scale", "3", "--seed", "1")),
    @("interval10_seed1", @("--scale", "10", "--interval", "0", "50", "--seed", "1"))
)
foreach ($r in $runs) {
    $name = $r[0]; $extra = $r[1]
    & $py generate_plus.py --out_dir "..\gen\$name" --offload @extra *> "..\gen_$name.log"
    "$name gen exit $LASTEXITCODE"
    & $py run_eval.py "..\gen\$name" *> "..\eval_$name.log"
    "$name eval exit $LASTEXITCODE"
}

$dirs = @("..\eval\gt", "..\gen\cfg4", "..\gen\cfg7", "..\gen\cfg10", "..\gen\cfg13", "..\gen\apg10", "..\gen\apg15",
          "..\gen\pe_cfg7", "..\gen\neg_cfg7", "..\gen\interval10", "..\gen\compose7_3", "..\gen\cfg10_50steps",
          "..\gen\cfg7_seed1", "..\gen\cfg10_seed1", "..\gen\compose7_3_seed1", "..\gen\interval10_seed1")
& $py mqascore.py @dirs *> ..\mqa_all.log
"mqa exit $LASTEXITCODE"
& $py mqascore.py ..\eval\gt ..\gen\cfg10 --shuffle *> ..\mqa_shuffle.log
"mqa shuffle exit $LASTEXITCODE"
Get-Content ..\eval\mqascore.csv

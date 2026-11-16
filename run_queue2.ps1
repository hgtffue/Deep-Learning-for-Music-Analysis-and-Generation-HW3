# Extra-credit experiments: run strictly one heavy job at a time (RAM limits), evaluate each run right after it.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = 1
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$py = "$env:USERPROFILE\miniconda3\envs\deepmir_hw3\python.exe"
Set-Location $PSScriptRoot

$runs = @(
    @("interval10",    @("--scale", "10", "--interval", "0", "50")),
    @("compose7_3",    @("--scale", "7", "--concepts", "concepts.csv", "--concept_scale", "3")),
    @("cfg10_seed1",   @("--scale", "10", "--seed", "1")),
    @("cfg10_50steps", @("--scale", "10", "--steps", "50"))
)
foreach ($r in $runs) {
    $name = $r[0]; $extra = $r[1]
    & $py generate_plus.py --out_dir "..\gen\$name" --offload @extra *> "..\gen_$name.log"
    "$name gen exit $LASTEXITCODE"
    & $py run_eval.py "..\gen\$name" *> "..\eval_$name.log"
    "$name eval exit $LASTEXITCODE"
}
Get-Content ..\eval\results.csv

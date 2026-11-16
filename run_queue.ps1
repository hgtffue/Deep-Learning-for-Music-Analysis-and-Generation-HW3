# Runs the improvement experiments sequentially after the baseline finishes, then evaluates everything.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = 1
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$py = "$env:USERPROFILE\miniconda3\envs\deepmir_hw3\python.exe"
Set-Location $PSScriptRoot

while ((Get-ChildItem ..\gen\cfg7\*.wav -ErrorAction SilentlyContinue).Count -lt 100) { Start-Sleep 30 }
Start-Sleep 20

$neg = "low quality, noisy, distorted, muffled, amateur recording, silence, vocals, speech"
$runs = @(
    @("cfg4",     @("--scale", "4")),
    @("cfg10",    @("--scale", "10")),
    @("apg10",    @("--guidance", "apg", "--scale", "10")),
    @("pe_cfg7",  @("--csv", "prompts_rewritten.csv", "--col", "prompt", "--scale", "7")),
    @("neg_cfg7", @("--negative", $neg, "--scale", "7"))
)
foreach ($r in $runs) {
    $name = $r[0]; $extra = $r[1]
    & $py generate_plus.py --out_dir "..\gen\$name" --offload @extra *> "..\gen_$name.log"
}

& $py run_eval.py ..\gen\cfg7 ..\gen\cfg4 ..\gen\cfg10 ..\gen\apg10 ..\gen\pe_cfg7 ..\gen\neg_cfg7 *> ..\eval.log
Get-Content ..\eval\results.csv

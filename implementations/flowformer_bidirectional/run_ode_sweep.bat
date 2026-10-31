@echo off
setlocal

if "%~4"=="" (
    echo Usage:
    echo   run_ode_sweep.bat TAG CKPT N T
    echo.
    echo Example:
    echo   run_ode_sweep.bat flowformer-mi_2d-reacdiff flowformer-mi_2d-reacdiff/flowformer-mi_2d-reacdiff_2000.pth 100 101
    exit /b 1
)

set "TAG=%~1"
set "CKPT=%~2"
set "N=%~3"
set "T=%~4"

set /a NUM_SAMPLES=N*(T-1)

if not exist "results\ode_inference_logs" (
    mkdir "results\ode_inference_logs"
)

echo ======================================================
echo Full-test ODE sweep
echo TAG          = %TAG%
echo Checkpoint   = %CKPT%
echo Trajectories = %N%
echo Frames       = %T%
echo NSP pairs    = %NUM_SAMPLES%
echo ======================================================

for %%O in (1 5 20 50 100) do (
    echo.
    echo ======================================================
    echo %TAG% ^| ODE steps = %%O
    echo ======================================================

    set "LOGFILE=results\ode_inference_logs\%TAG%_ode%%O.log"

    powershell -NoProfile -Command ^
        "& python -u infer_flowformer.py --ckpt $env:CKPT --num_ode_steps %%O --max_T_rollout %T% --num_traj_rollout %N% --num_samples %NUM_SAMPLES% --cyclic_runs 1 2>&1 | Tee-Object -FilePath $env:LOGFILE; exit $LASTEXITCODE"

    if errorlevel 1 (
        echo.
        echo ERROR: ODE %%O failed.
        exit /b 1
    )
)

echo.
echo ======================================================
echo Finished: %TAG%
echo Logs stored in:
echo results\ode_inference_logs\%TAG%_ode*.log
echo ======================================================

endlocal

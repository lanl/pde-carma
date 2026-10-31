@echo off
setlocal

echo ======================================================
echo ReacDiff full-test ODE ablations
echo N = 100 trajectories
echo T = 101 frames
echo ======================================================

echo.
echo ======================================================
echo ReacDiff 3.5M
echo ======================================================

call run_ode_sweep.bat ^
    flowformer-mi_2d-reacdiff ^
    flowformer-mi_2d-reacdiff_2000.pth ^
    100 ^
    101

if errorlevel 1 exit /b 1

echo.
echo ======================================================
echo ReacDiff 10M
echo ======================================================

call run_ode_sweep.bat ^
    flowformer-s_2d-reacdiff ^
    flowformer-s_2d-reacdiff_2000.pth ^
    100 ^
    101

if errorlevel 1 exit /b 1

echo.
echo ======================================================
echo ReacDiff 25M
echo ======================================================

call run_ode_sweep.bat ^
    flowformer-m_2d-reacdiff ^
    flowformer-m_2d-reacdiff_2000.pth ^
    100 ^
    101

if errorlevel 1 exit /b 1

echo.
echo ======================================================
echo ALL REACDIFF ODE SWEEPS FINISHED
echo ======================================================

endlocal

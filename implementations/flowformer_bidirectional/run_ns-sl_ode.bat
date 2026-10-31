@echo off
setlocal

echo ======================================================
echo FNS-KF full-test ODE ablations
echo N = 240 trajectories
echo T = 21 frames
echo ======================================================

@REM  echo.
@REM  echo ======================================================
@REM  echo NS-SL 3.5M
@REM  echo ======================================================

@REM  call run_ode_sweep.bat ^
@REM      flowformer-mi_ns-sl ^
@REM      flowformer-mi_ns-sl_1335.pth ^
@REM      240 ^
@REM      21

@REM  if errorlevel 1 exit /b 1

@REM  echo.
@REM  echo ======================================================
@REM  echo NS-SL 10M
@REM  echo ======================================================

@REM  call run_ode_sweep.bat ^
@REM      flowformer-s_ns-sl ^
@REM      flowformer-s_ns-sl_1335.pth ^
@REM      240 ^
@REM      21

@REM  if errorlevel 1 exit /b 1

echo.
echo ======================================================
echo NS-SL 25M
echo ======================================================

call run_ode_sweep.bat ^
    flowformer-m_ns-sl ^
    flowformer-m_ns-sl_1335.pth ^
    240 ^
    21

if errorlevel 1 exit /b 1

endlocal

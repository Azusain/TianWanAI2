@echo off
echo Starting TianWan AI Detection Services...

echo.
echo [1/3] Starting Fire Detection Service (Port 8092)...
start "Fire Service" cmd /c "python fire_service.py"

echo [2/3] Starting Helmet Safety Service (Port 8093)...  
start "Helmet Service" cmd /c "python helmet_service.py"

echo [3/3] Starting API Gateway (Port 8091)...
timeout /t 5 /nobreak >nul
start "API Gateway" cmd /c "python gateway.py"

echo.
echo All services starting up...
echo - Fire Detection: http://localhost:8092
echo - Helmet Safety: http://localhost:8093  
echo - API Gateway: http://localhost:8091
echo.
echo Available endpoints:
echo - POST http://localhost:8091/fire
echo - POST http://localhost:8091/helmet
echo - GET http://localhost:8091/models
echo.
echo Press any key to continue...
pause >nul

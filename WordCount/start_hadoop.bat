@REM 分别启动新终端执行以下命令来启动Hadoop服务

@echo Starting Hadoop services...

start cmd /k %HADOOP_HOME%\sbin\start-dfs.cmd
%HADOOP_HOME%\sbin\start-yarn.cmd

@REM 最后一个脚本会关掉终端窗口


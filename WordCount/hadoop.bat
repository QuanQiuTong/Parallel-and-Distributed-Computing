@REM 在 hadoop 中操作

hadoop fs -mkdir -p /input
hadoop fs -put -f ./1k.txt /input
hadoop jar wordcount.jar edu.fudan.WordCount /input /output
hadoop fs -cat /output/part-r-00000
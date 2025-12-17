hadoop jar %HADOOP_HOME%/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.4.2.jar ^
  randomtextwriter ^
  -D mapreduce.randomtextwriter.totalbytes=100000000 ^
  -D mapreduce.randomtextwriter.bytespermap=10000000 ^
  /input/100m

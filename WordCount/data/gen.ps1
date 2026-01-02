hadoop jar %HADOOP_HOME%/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.4.2.jar ^
  randomtextwriter ^
  -D mapreduce.randomtextwriter.totalbytes=500000000 ^
  -D mapreduce.randomtextwriter.bytespermap=10000000 ^
  /input/500m

hadoop jar %HADOOP_HOME%/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.4.2.jar ^
  randomtextwriter ^
  -D mapreduce.randomtextwriter.totalbytes=1000000000 ^
  -D mapreduce.randomtextwriter.bytespermap=10000000 ^
  /input/1000m

hadoop jar %HADOOP_HOME%/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.4.2.jar ^
  randomtextwriter ^
  -D mapreduce.randomtextwriter.totalbytes=10000000 ^
  -D mapreduce.randomtextwriter.bytespermap=10000000 ^
  /input/1m

hadoop jar %HADOOP_HOME%/share/hadoop/mapreduce/hadoop-mapreduce-client-jobclient-*-tests.jar TestDFSIO -write -nrFiles 1 -fileSize 1024

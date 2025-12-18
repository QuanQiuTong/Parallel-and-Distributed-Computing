package edu.fudan;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class WordCount extends Configured implements Tool {
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new Configuration(), new WordCount(), args);
        System.exit(exitCode);
    }

    @Override
    public int run(String[] args) throws Exception {
        // 假设 args[1] 是输入目录的根目录 /input
        Path baseInputPath = new Path(args[1]);
        Path baseOutputPath = new Path(args[2]);
        
        // 定义要测试的文件列表
        String[] fileNames = {"wordcount_1k.txt", "wordcount_1m.txt", "wordcount_10m.txt", "wordcount_100m.txt", "wordcount_1000m.txt"};

        for (String fileName : fileNames) {
            System.out.println("===========================================");
            System.out.println("Testing Data Size: " + fileName);

            Path specificInput = new Path(baseInputPath, fileName);
            Path specificOutput = new Path(baseOutputPath, fileName);

            for (int numReducers : new int[] { 1, 2, 4 }) {
                for(int numMapTasks : new int[] {1, 2, 4, 8, 16}) {
                    Job job = getJob(specificInput, specificOutput, numReducers, numMapTasks);
                    runJob(job);
                }
            }
        }
        return 0;
    }

    private Job getJob(Path inputPath, Path outputPath, int numReducers, int numMapTasks) throws Exception {
        Configuration conf = getConf();
        // System.out.println("framework = " + conf.get("mapreduce.framework.name"));

        conf.set("mapreduce.framework.name", "local");
        conf.setInt("mapreduce.local.map.tasks.maximum", numMapTasks);
        conf.setInt("mapreduce.task.io.sort.mb", 50);
        
        conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 16 * 1024 * 1024); // 16MB
        conf.setLong("mapreduce.input.fileinputformat.split.minsize", 16 * 1024 * 1024);

        Job job = Job.getInstance(conf, numReducers + " reducers with " + numMapTasks + " map tasks");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(WordCountMapper.class);
        job.setCombinerClass(WordCountReducer.class);
        job.setReducerClass(WordCountReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // 设置Reducer任务数量
        job.setNumReduceTasks(numReducers);

        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(outputPath)) {
            fs.delete(outputPath, true);
        }
        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputPath);

        return job;
    }

    private void runJob(Job job) throws Exception {
        long startTime = System.currentTimeMillis();
        boolean success = job.waitForCompletion(true);
        long endTime = System.currentTimeMillis();
        if (success) {
            System.out.println("Job " + job.getJobName() + " Completed Successfully in " + (endTime - startTime) + " milliseconds");
        } else {
            System.out.println("Job " + job.getJobName() + " Failed");
        }
        System.out.println("-------------------------------------------");
    }
}

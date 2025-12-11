package edu.fudan;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

    public static void main(String[] args) throws Exception {
        Path inputPath = new Path(args[0]);
        Path outputPath = new Path(args[1]);

        int[] reducerCounts = {1, 2, 4, 8};

        for (int i: reducerCounts) {
            Job job = getJob(inputPath, new Path(outputPath, "run_" + i), i);
            long startTime = System.currentTimeMillis();

            boolean success = job.waitForCompletion(true);

            long endTime = System.currentTimeMillis();
            long executionTime = endTime - startTime;

            System.out.println("==========================================");
            System.out.println("作业配置: " + job.getNumReduceTasks() + " 个Reducer");
            System.out.println("输入路径: " + inputPath.toString());
            System.out.println("输出路径: " + job.getConfiguration().get("mapreduce.output.fileoutputformat.outputdir"));
            System.out.println("执行时间: " + executionTime + " ms");
            System.out.println("==========================================");

            if (!success) {
                System.exit(1);
            }
        }

        System.exit(0);
    }

    private static Job getJob(Path inputPath, Path outputPath, int numReducers) throws Exception {
        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf, "word count with " + numReducers + " reducers");
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
}

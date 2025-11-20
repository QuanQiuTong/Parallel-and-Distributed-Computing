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
        if (args.length < 3) {
            System.err.println("Usage: WordCount <input_path> <output_path> <num_reducers>");
            System.err.println("Example: WordCount /input /output 4");
            System.exit(1);
        }

        Path inputPath = new Path(args[0]);
        Path outputPath = new Path(args[1]);
        int numReducers = Integer.parseInt(args[2]);

        Job job = WordCount.getJob(inputPath, outputPath, numReducers);

        // 记录开始时间
        long startTime = System.currentTimeMillis();

        boolean success = job.waitForCompletion(true);

        // 记录结束时间并输出
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;

        System.out.println("==========================================");
        System.out.println("作业配置: " + numReducers + " 个Reducer");
        System.out.println("输入路径: " + inputPath.toString());
        System.out.println("输出路径: " + outputPath.toString());
        System.out.println("执行时间: " + executionTime + " ms");
        System.out.println("==========================================");

        System.exit(success ? 0 : 1);
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

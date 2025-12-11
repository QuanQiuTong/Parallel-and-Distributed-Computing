import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

public class SingleThreadWordCount {
    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: SingleThreadWordCount <input_file> <output_file>");
            System.exit(1);
        }
        
        long startTime = System.currentTimeMillis();
        
        Map<String, Integer> wordCount = new HashMap<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(args[0]))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] words = line.split("\\s+");
                for (String word : words) {
                    word = word.trim();
                    if (!word.isEmpty()) {
                        wordCount.put(word, wordCount.getOrDefault(word, 0) + 1);
                    }
                }
            }
        }
        
        try (PrintWriter writer = new PrintWriter(args[1])) {
            for (Map.Entry<String, Integer> entry : wordCount.entrySet()) {
                writer.println(entry.getKey() + "\t" + entry.getValue());
            }
        }
        
        long endTime = System.currentTimeMillis();
        System.out.println("Single Thread Time Usage: " + (endTime - startTime) + " ms");
    }
}
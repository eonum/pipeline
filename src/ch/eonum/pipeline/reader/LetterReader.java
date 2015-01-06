package ch.eonum.pipeline.reader;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.util.HashMap;

import ch.eonum.pipeline.core.DenseSequence;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.SequenceDataSet;

public class LetterReader {

	/**
	 * 
	 * @param filename
	 * @param predictionAfter begin prediction after X letters
	 * @param lineByLine make one sequence for each line
	 * @param addNewLine add line breaks as letters
	 * @return
	 * @throws IOException
	 * @throws ParseException
	 */
	public static SequenceDataSet<DenseSequence> readTraining(String filename, boolean lineByLine, 
			boolean addNewLine, Features features) throws IOException,
			ParseException {
		SequenceDataSet<DenseSequence> data = new SequenceDataSet<DenseSequence>();
		
		FileInputStream fstream = new FileInputStream(filename);

		DataInputStream in = new DataInputStream(fstream);
		BufferedReader br = new BufferedReader(new InputStreamReader(in));
		
		String line;
				
		DenseSequence seq = createSequence(features);
		
		int lineNumber = 0;
		
		while ((line = br.readLine()) != null) {
			line = line.trim().toLowerCase();
			if(line.length() <= 1)
				continue;
			
			for(int i = 0; i < line.length(); i++){
				String c = "" + line.charAt(i);
				if(features.hasFeature(c)){
					double[] point = new double[features.size()];
					point[features.getIndexFromFeature(c)] = 1.0;
					seq.addTimePoint(point);
				}
			}
			
			if(addNewLine && features.hasFeature("\n")){
				double[] point = new double[features.size()];
				point[features.getIndexFromFeature("\n")] = 1.0;
				seq.addTimePoint(point);
			}	
			
			lineNumber++;
			if(lineByLine){
				seq.id = "line" + lineNumber;
				data.addInstance(seq);
				seq = createSequence(features);
			}	
		}
		
		if(!lineByLine)
			data.addInstance(seq);
		
		in.close();
		
		return data;
	}

	private static DenseSequence createSequence(Features features) {
		DenseSequence seq = new DenseSequence("dataset", "all",
				new HashMap<String, Double>(), features);
		return seq;
	}
}

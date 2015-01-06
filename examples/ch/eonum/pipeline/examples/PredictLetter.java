package ch.eonum.pipeline.examples;


import java.io.IOException;
import java.text.ParseException;

import ch.eonum.pipeline.classification.lstm.LSTMClassifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.DenseSequence;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.SequenceDataSet;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.evaluation.RecognitionRateSequence;
import ch.eonum.pipeline.reader.LetterReader;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.validation.SystemValidator;

/**
 * Toy experiment for the LSTM (Long Short Term Memory Recurrent Neural Network)
 * using artificial letter sequences. You can test the net by inventing your own
 * letter patterns and by tweaking the meta parameters.
 * 
 * @author tim
 * 
 */
public class PredictLetter {
	public static final String dataset = "examples/results/letters.txt";
	public static final String resultsFolder = "examples/results/letters/";
	public static final char[] allowedChars = new char[] { 'a', 'b', 'c', 'd', 'f', '\n'};/*,
			'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
			'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'ü', '.',
			',', ':', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
			'9', ' '};*/

	/**
	 * Test Validation Script for the evaluation of models. Execute with enough
	 * memory: -Xmx1024m
	 * 
	 * @param args
	 * @throws IOException 
	 * @throws ParseException 
	 */
	public static void main(String[] args) throws IOException, ParseException {
		FileUtil.mkdir(resultsFolder);
		
		Features allowedCharMap = new Features();
		for(int i = 0; i < allowedChars.length; i++)
			allowedCharMap.addFeature("" + allowedChars[i]);
		allowedCharMap.recalculateIndex();
		
		SequenceDataSet<DenseSequence> dataTraining = LetterReader.readTraining(dataset, true, false, allowedCharMap);
		dataTraining.setTimeLag(1, allowedCharMap);
			
		Features dims = Features.createFromDataSets(dataTraining);
		
		dims.writeToFile(resultsFolder + "features.txt");
	
		DataSet<DenseSequence> dataValidation = dataTraining.extractSubSet(0.5);
		Evaluator<DenseSequence> recRate = new RecognitionRateSequence<DenseSequence>();
		
		LSTMClassifier<DenseSequence> lstm = new LSTMClassifier<DenseSequence>();
		lstm.setClasses(dims);
		lstm.setForgetGateUse(false);
		lstm.setInputGateUse(true);
		lstm.setOutputGateUse(true);
		lstm.setFeatures(dims);
		lstm.setBaseDir(resultsFolder + "lstm/");
		FileUtil.mkdir(resultsFolder + "lstm/");
		
		
		lstm.putParameter("numNets", 1.0);
		lstm.putParameter("numNetsTotal", 1.0);
		lstm.putParameter("numLSTM", 2.0);
		lstm.putParameter("memoryCellBlockSize", 2.0);
		lstm.putParameter("numHidden", 0.0);
		lstm.putParameter("maxEpochsAfterMax", 50);
		lstm.putParameter("maxEpochs", 1000);
		lstm.putParameter("learningRate", 0.01);
		lstm.putParameter("batchSize", 1.0);
		lstm.putParameter("momentum", 0.1);
		

		lstm.setTestSet(dataValidation);
		lstm.setTrainingSet(dataTraining);
		SystemValidator<DenseSequence> lstmSystem = new SystemValidator<DenseSequence>(lstm, recRate);
		lstmSystem.setBaseDir(resultsFolder);	
		
		lstmSystem.evaluate(true, "letters-lstm");
		System.out.println("Optimum: " + recRate.evaluate(dataValidation));
	}
		
}

package ch.eonum.pipeline.examples;


import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

import ch.eonum.pipeline.classification.lstm.LSTMClassifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.DenseSequence;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Parameters;
import ch.eonum.pipeline.core.SequenceDataSet;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.evaluation.RecognitionRateSequence;
import ch.eonum.pipeline.reader.LetterReader;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.validation.ParameterValidation;
import ch.eonum.pipeline.validation.SystemValidator;

public class PredictLetter {
	public static final String dataset = "data-lstm-letter/sequence.txt";
	public static final String resultsFolder = "data-lstm-letter/bookletter-lines/";
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
		
//		Map<Character, Integer> charCount = new HashMap<Character, Integer>();
//		for(int i = 0; i < allowedChars.length; i++)
//			charCount.put(allowedChars[i], 0);
//		int total = 0;
//		for(Instance e : dataTraining){
//			Sequence s = (Sequence)e;
//			total += s.getSequenceLength();
//			for(Character c : allowedChars){
//				for(int i = 0; i < s.getSequenceLength(); i++)
//					if(s.get(i, "" + c) > 0.5)
//						charCount.put(c, charCount.get(c) + 1);
//			}
//		}
//		for(Character c : charCount.keySet())
//			System.out.println("Frequency for " + c + ": " + charCount.get(c)/(double)total);
//		System.exit(0);
			
		Features dims = Features.createFromDataSets(dataTraining);
		
		dims.writeToFile(resultsFolder + "features.txt");
	
		DataSet<DenseSequence> dataValidation = dataTraining;
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
		lstm.putParameter("numLSTM", 8.0);
		lstm.putParameter("memoryCellBlockSize", 2.0);
		lstm.putParameter("numHidden", 0.0);
		lstm.putParameter("maxEpochsAfterMax", 1000);
		lstm.putParameter("maxEpochs", 1000);
		lstm.putParameter("learningRate", 0.01);
		lstm.putParameter("batchSize", 1.0);
		lstm.putParameter("momentum", 0.1);
		

		lstm.setTestSet(dataValidation);
		lstm.setTrainingSet(dataTraining);
		SystemValidator<DenseSequence> lstmSystem = new SystemValidator<DenseSequence>(lstm, recRate);
		lstmSystem.setBaseDir(resultsFolder);
		
		List<ParameterValidation> paramsGradientAscent = new ArrayList<ParameterValidation>();
		
		paramsGradientAscent.add(new ParameterValidation(new Parameters[] {
				lstm }, "numLSTM", 1.0, 8.0, 1.0,
				20.0, 2.0, 1.0, false));
		paramsGradientAscent.add(new ParameterValidation(new Parameters[] {
				lstm }, "learningRate", -14, -2, -8,
				0.0, 0.01, 1.0, true));
		paramsGradientAscent.add(new ParameterValidation(new Parameters[] {
				lstm }, "memoryCellBlockSize", 1.0, 8.0, 1.0,
				20.0, 1.0, 1.0, false));
		

//		Map<ParameterValidation, Double> params = lstmSystem.gradientAscent(paramsGradientAscent, 5, resultsFolder + "parameter_validation/");
//		Log.puts("Optimal Parameters: " + params);
//		ParameterValidation.updateParameters(params);		
		
		lstmSystem.evaluate(true, "nn-all");
		System.out.println("Optimum: " + recRate.evaluate(dataValidation));

	}
		
}

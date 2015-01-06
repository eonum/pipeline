package ch.eonum.pipeline.examples;


import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
import ch.eonum.pipeline.util.Log;
import ch.eonum.pipeline.validation.ParameterValidation;
import ch.eonum.pipeline.validation.SystemValidator;

/**
 * This is an example of a Long Short Term Memory recurrent neural network
 * trying to learn Java. The network tries to predict the next character in a
 * Java source code file.
 * 
 * @author tim
 * 
 */
public class PredictCode {
	/** An arbitrary Java source code file providing training, validation and test data. */
	public static final String dataset = "src/ch/eonum/pipeline/core/DenseSequence.java";
	/** folder where results (the model, validation results, gnuplots) will be written to. */
	public static final String resultsFolder = "examples/results/javacode/";
	/** allowed tokens/characters. */
	public static final char[] allowedChars = new char[] { 'a', 'b', 'c', 'd',
			'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
			'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.', ',', ':', '-',
			'/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '{',
			'}', '(', ')', '+', '[', ']', ';', '=', '<', '>', '\\', '*', '"', '\n'};

	/**
	 * Execute with enough memory: -Xmx1024m
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
		
		SequenceDataSet<DenseSequence> dataTraining = LetterReader.readTraining(dataset, true, true, allowedCharMap);
		dataTraining.setTimeLag(1, allowedCharMap);
		
		Features dims = Features.createFromDataSets(dataTraining);
		
		dims.writeToFile(resultsFolder + "features.txt");
	
		DataSet<DenseSequence> dataValidation = dataTraining;
		Evaluator<DenseSequence> recRate = new RecognitionRateSequence<DenseSequence>();
		
		LSTMClassifier<DenseSequence> lstm = new LSTMClassifier<DenseSequence>();
		lstm.setClasses(dims);
		lstm.setForgetGateUse(true);
		lstm.setInputGateUse(true);
		lstm.setOutputGateUse(true);
		lstm.setFeatures(dims);
		lstm.setBaseDir(resultsFolder + "lstm/");
		FileUtil.mkdir(resultsFolder + "lstm/");
		
		/** parameterization of the LSTM. */
		lstm.putParameter("numNets", 1.0);
		lstm.putParameter("numNetsTotal", 1.0);
		lstm.putParameter("numLSTM", 4.0);
		lstm.putParameter("memoryCellBlockSize", 2.0);
		lstm.putParameter("numHidden", 0.0);
		lstm.putParameter("maxEpochsAfterMax", 50);
		lstm.putParameter("maxEpochs", 55);
		lstm.putParameter("learningRate", 0.011);
		lstm.putParameter("batchSize", 1.0);
		lstm.putParameter("momentum", 0.);
		
		lstm.setTestSet(dataValidation);
		lstm.setTrainingSet(dataTraining);
		SystemValidator<DenseSequence> lstmSystem = new SystemValidator<DenseSequence>(lstm, recRate);
		lstmSystem.setBaseDir(resultsFolder);
		
		/** validation of meta parameters. */
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
		

		Map<ParameterValidation, Double> params = lstmSystem.gradientAscent(paramsGradientAscent, 5, resultsFolder + "parameter_validation/");
		Log.puts("Optimal Parameters: " + params);
		ParameterValidation.updateParameters(params);		
		
		long millis = System.currentTimeMillis();
		
		/** final classification. */
		lstmSystem.evaluate(true, "nn-all");
		System.out.println("Optimum: " + recRate.evaluate(dataValidation));
		
		System.out.println("Execution time: " + ((System.currentTimeMillis() - millis) / 1000.0) + " seconds");

	}
		
}

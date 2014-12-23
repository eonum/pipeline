package ch.eonum.pipeline.features;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataPipeline;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Parameters;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.util.Log;

/**
 * Feature selection class.
 * select the best features.
 * create a ranked list of all features.
 * 
 * @author tim
 *
 */
public class FeatureSelection<E extends Instance> extends Parameters implements DataPipeline<E> {
	private static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("mode", "mode of feature removing. 'nbest' or 'negative' (default 'negative')");
		PARAMETERS.put("nbest", "number of features which are not removed in nbest mode (default: 100)");
	}

	private Classifier<E> classifier;
	private Features features;
	private Evaluator<E> evaluator;
	private List<FeatureDelta> featureDeltas;
	private DataSet<E> trainingData;
	private DataSet<E> testData;
	private DataPipeline<E> inputTraining;
	private DataPipeline<E> inputTest;
	private boolean doRetrain;

	public FeatureSelection(Classifier<E> classifier, Features dims, Evaluator<E> eval, DataSet<E> training, DataSet<E> test) {
		this.classifier = classifier;
		this.features = dims;
		this.evaluator = eval;
		this.trainingData = training;
		this.testData = test;
		this.setSupportedParameters(FeatureSelection.PARAMETERS);
		this.putParameter("mode", "negative");
		this.putParameter("nbest", 100.0);
		this.doRetrain = true;
	}

	/**
	 * create and save a ranked list with each feature
	 * @param string
	 */
	public void createAndSaveRankedList(String fileName) {
		this.createRankedList();
		this.saveRankedList(fileName);
	}
	
	/**
	 * create and save feature frequency ranking
	 * @param string
	 */
	public void createAndSaveRankedFrequencyList(String fileName) {
		this.createFrequencyList();
		this.saveRankedList(fileName);
	}
	
	/**
	 * create and save feature correlation ranking
	 * @param string
	 */
	public void createAndSaveRankedCorrelationList(String fileName) {
		this.createCorrelationList();
		this.saveRankedList(fileName);
	}

	/**
	 * rate the features according their frequency in the training set.
	 */
	public void createFrequencyList() {
		this.featureDeltas = new ArrayList<FeatureDelta>();

		for(int i = 0; i < features.size(); i++){
			double sum = 0.0;
			String feature = this.features.getFeatureByIndex(i);
			Log.puts("Processing feature: " + feature + " (" + i + ")");
			for(Instance each : this.trainingData)
				sum += each.get(feature);
			this.featureDeltas.add(new FeatureDelta(feature, sum));
		}
		
		Collections.sort(this.featureDeltas);
	}
	
	/**
	 * rank each feature according their absolute correlation with the groundtruth
	 */
	public void createCorrelationList() {
		this.featureDeltas = new ArrayList<FeatureDelta>();
		
		double avgGt = 0.0;
		for(Instance each : this.trainingData)
			avgGt += each.groundTruth.equals("1") ? 1.0 : 0.0;
		avgGt /= this.trainingData.size();
		
		double sumSquareDeltasGt = 0.0;
		for(Instance each : this.trainingData)
			sumSquareDeltasGt += Math.pow(((each.groundTruth.equals("1") ? 1.0 : 0.0) - avgGt), 2);
		

		for(int i = 0; i < features.size(); i++){
			String feature = this.features.getFeatureByIndex(i);
			Log.puts("Processing feature: " + feature + " (" + i + ")");
			
			double avg = 0.0;
			for(Instance each : this.trainingData)
				avg += each.get(feature);
			avg /= this.trainingData.size();
			
			double covariance = 0.0;
			double variance = 0.0;
			for(Instance each : this.trainingData){
				double fDelta = each.get(feature) - avg;
				double gtDelta = (each.groundTruth.equals("1") ? 1.0 : 0.0) - avgGt;
				covariance += fDelta * gtDelta;
				variance += Math.pow(fDelta, 2);
			}
			
			double corr = covariance / (Math.sqrt(variance*sumSquareDeltasGt));
			if(Double.isNaN(corr)) corr = 0.0;
			this.featureDeltas.add(new FeatureDelta(feature, Math.abs(corr)));
		}
		
		Collections.sort(this.featureDeltas);
	}

	/**
	 * load a ranked list from file.
	 * recreating the list can be avoided.
	 * 
	 * @param fileName
	 */
	public void loadRankedList(String fileName){
		this.featureDeltas = new ArrayList<FeatureDelta>();
		try {

			FileInputStream fstream = new FileInputStream(fileName);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			while ((strLine = br.readLine()) != null) {
				String[] values = strLine.split(";");
				this.featureDeltas.add(new FeatureDelta(values[0], Double.valueOf(values[1])));
			}
			
			in.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * save the ranked feature list to a file.
	 * @param fileName
	 */
	public void saveRankedList(String fileName) {
		try {
			PrintStream p = new PrintStream(new FileOutputStream(fileName));
			for(FeatureDelta fd : this.featureDeltas){
				String feature = fd.getFeature();
				p.println(feature + ";" + fd.getDelta() + ";" + this.features.getDescription(feature));
			}
			
			p.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * create the ranked list.
	 * calculate for each feature the difference 
	 */
	public void createRankedList() {
		this.featureDeltas = new ArrayList<FeatureDelta>();
		// reference measure
		double reference = this.test(trainingData, testData, true);
		
		DataSet<E> training = this.trainingData.deepCopy();
		DataSet<E> test = this.testData.deepCopy();

		List<String> f = features.getListOfFeaturesCopy();
		for(int i = 0; i < f.size(); i++){
			String feature = f.get(i);
			Log.puts("Processing feature: " + feature + " (" + i + ")");

			training.removeFeature(feature);
			test.removeFeature(feature);
			features.removeFeature(feature);
			features.recalculateIndex();
			double result = this.test(training, test, this.doRetrain);
			features.addFeature(feature);
			features.recalculateIndex();
			this.featureDeltas.add(new FeatureDelta(feature, reference - result));
			
			training = this.trainingData.deepCopy();
			test = this.testData.deepCopy();
		}
		
		Collections.sort(this.featureDeltas);
	}
	
	/**
	 * remove all features which have a negative effect on the result.
	 * @param trainset
	 */
	public void removeNegative(DataSet<? extends Instance> data) {
		for(FeatureDelta each : this.featureDeltas)
			if(each.getDelta() <= 0){
				data.removeFeature(each.getFeature());
				this.features.removeFeature(each.getFeature());
			} else {
				this.features.addFeature(each.getFeature());
			}
		
		this.features.recalculateIndex();
	}
	
	/**
	 * remove all features except the n-best.
	 * @param trainset
	 */
	public void removeNBest(DataSet<? extends Instance> data, int n) {
		int i = 0;
		for(FeatureDelta each : this.featureDeltas){
			i++;
			if(i > n) {
				data.removeFeature(each.getFeature());
				this.features.removeFeature(each.getFeature());
			} else {
				this.features.addFeature(each.getFeature());
			}
		}
		this.features.recalculateIndex();
	}
	
	/**
	 * remove features according the mode parameter
	 * @param set
	 * @return
	 */
	public DataSet<E> removeFeatures(DataSet<E> set) {
		if(this.getStringParameter("mode").equals("negative"))
			this.removeNegative(set);
		else if(this.getStringParameter("mode").equals("nbest"))
			this.removeNBest(set, (int)this.getDoubleParameter("nbest"));
		else 
			Log.error("Unsupported mode " + this.getStringParameter("mode"));
		return set;
	}
	
	/**
	 * do not retrain when creating the list.
	 * this is not the best way to obtain the list. But it is much faster.
	 */
	public void setNoRetrain() {
		this.doRetrain = false;
	}


	private double test(DataSet<E> train, DataSet<E> test, boolean doTraining) {
		if(doTraining){
			this.classifier.setTrainingSet(train);
			this.classifier.train();
		}	
		this.classifier.setTestSet(test);
		return this.evaluator.evaluate(classifier.test());
	}
	
	/** pipeline methods. */
	
	@Override
	public DataSet<E> trainSystem(boolean isResultDataSetNeeded) {
		DataSet<E> set;
		if(this.inputTraining != null)
			set = this.inputTraining.trainSystem(isResultDataSetNeeded);
		else
			set = this.trainingData.deepCopy();
		return this.removeFeatures(set);
	}

	@Override
	public DataSet<E> testSystem() {
		DataSet<E> set;
		if(this.inputTest != null)
			set = this.inputTest.testSystem();
		else
			set = this.testData.deepCopy();
		return this.removeFeatures(set);
	}

	@Override
	public void addInputTraining(DataPipeline<E> input) {
		this.inputTraining = input;
	}

	@Override
	public void addInputTest(DataPipeline<E> input) {
		this.inputTest = input;
	}

	public void reduceFeatures(Features features) {
		int i = 0;
		for(FeatureDelta each : this.featureDeltas){
			i++;
			if(i > this.getDoubleParameter("nbest")) {
				features.removeFeature(each.getFeature());
			} 
		}
		features.recalculateIndex();
	}

}

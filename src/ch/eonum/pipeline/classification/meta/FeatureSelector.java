package ch.eonum.pipeline.classification.meta;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.features.FeatureDelta;
import ch.eonum.pipeline.util.Log;

/**
 * plus-l minus-r forward search feature selection algorithm.
 * Non greedy.
 * 
 * @author tim
 *
 */
public class FeatureSelector<E extends Instance> extends Classifier<E> {

	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("l", "number of features to add in each iteration. (default: 5.0)");
		PARAMETERS.put("r", "number of features to remove in each iteration. (default: 2.0)");
	}

	private Classifier<E> baseClassifier;
	private Features reducedFeatures;
	private Evaluator<E> evaluator;
	private PrintWriter log;

	public FeatureSelector(Classifier<E> baseClassifier, Features features, Evaluator<E> evaluator) {
		this.evaluator = evaluator;
		this.baseClassifier = baseClassifier;
		this.features = features;
		this.testDataSet = this.baseClassifier.getTestDataSet();
		this.trainingDataSet = this.baseClassifier.getTrainingDataSet();
		this.setSupportedParameters(FeatureSelector.PARAMETERS);
		this.putParameter("l", 5.0);
		this.putParameter("r", 2.0);
	}

	@Override
	public void train() {
		try {
			log = new PrintWriter(this.baseDir + "featureReduction.log");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		this.baseClassifier.setTrainingSet(trainingDataSet);
		this.baseClassifier.setTestSet(testDataSet);
		int l = (int)this.getDoubleParameter("l");
		int r = (int)this.getDoubleParameter("r");
		this.reducedFeatures = new Features();
		baseClassifier.setFeatures(reducedFeatures);
		double max = Double.NEGATIVE_INFINITY;
		int iteration = 0;
		int afterMax = 0;
		Features maxFeatures = null;
		while(true){
			iteration++;
			ArrayList<FeatureDelta> ranking = this.createAddFeaturesRanking(features, reducedFeatures);
			for(int i = 0; i < l; i++){
				log("Adding feature: " + ranking.get(i).getFeature());
				reducedFeatures.addFeature(ranking.get(i).getFeature());
			}
			reducedFeatures.recalculateIndex();
			ranking = this.createRemoveFeaturesRanking(reducedFeatures);
			for(int i = 0; i < r; i++){
				log("Remove feature: " + ranking.get(i).getFeature());
				reducedFeatures.removeFeature(ranking.get(i).getFeature());
			}
			reducedFeatures.recalculateIndex();
			baseClassifier.train();
			double eval = this.evaluator.evaluate(this.baseClassifier.test());
			log("Iteration " + iteration + ": " + eval + " Size of reduced feature set: " + reducedFeatures.size());
			if(eval <= max)
				afterMax++;
			else {
				max = eval;
				afterMax = 0;
				maxFeatures = this.reducedFeatures.copy();
			}
			if(afterMax >= 5)
				break;
		}
		this.reducedFeatures = maxFeatures;
		this.reducedFeatures.writeToFile(baseDir + "reduced-features.txt");
		baseClassifier.setFeatures(reducedFeatures);
		baseClassifier.train();
		log.close();
	} 
	
	private void log(String message) {
		Log.puts(message);
		this.log.println(message);
		this.log.flush();
	}

	private ArrayList<FeatureDelta> createRemoveFeaturesRanking(
			Features reduced) {
		ArrayList<FeatureDelta> featureDeltas = new ArrayList<FeatureDelta>();

		for(String feature : reduced.getListOfFeaturesCopy()){
			reduced.removeFeature(feature);
			reduced.recalculateIndex();
			baseClassifier.train();
			double result = this.evaluator.evaluate(this.baseClassifier.test());
			reduced.addFeature(feature);
			reduced.recalculateIndex();
			featureDeltas.add(new FeatureDelta(feature, result));
		}
		
		Collections.sort(featureDeltas);
		this.reducedFeatures.writeToFile(baseDir + "reduced-features.txt");
		return featureDeltas;
	}

	public ArrayList<FeatureDelta> createAddFeaturesRanking(Features all, Features reduced) {
		ArrayList<FeatureDelta> featureDeltas = new ArrayList<FeatureDelta>();

		for(String feature : all.getListOfFeaturesCopy()){
			if(reduced.hasFeature(feature))
				continue;
			reduced.addFeature(feature);
			reduced.recalculateIndex();
			baseClassifier.train();
			double result = this.evaluator.evaluate(this.baseClassifier.test());
			reduced.removeFeature(feature);
			reduced.recalculateIndex();
			featureDeltas.add(new FeatureDelta(feature, result));
		}
		
		Collections.sort(featureDeltas);
		return featureDeltas;
	}

	@Override
	public DataSet<E> test() {
		baseClassifier.setTestSet(testDataSet);
		baseClassifier.setFeatures(reducedFeatures);
		return baseClassifier.test();
	}

}

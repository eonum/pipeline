package ch.eonum.pipeline.classification.meta;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import ch.eonum.pipeline.util.Log;
import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.FileUtil;

/**
 * This meta classifier creates several subclasses for each class (superclass).
 * The subclasses are created in a sort of K-Means algorithm.
 * 
 * Use this classifier if you assume that the data within one class is composed
 * of several different distinct patterns
 * 
 * @author tim
 *
 */
public class ClassifierClusterer<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("S", "number of subclasses for each (super)class (default 5.0)");
		PARAMETERS.put("k", "maximum number of iterations (default 5.0)");
	}

	private Classifier<E> baseClassifier;

	public ClassifierClusterer(Classifier<E> baseClassifier) {
		this.baseClassifier = baseClassifier;
		this.setSupportedParameters(ClassifierClusterer.PARAMETERS);
		this.putParameter("S", 6.0);
		this.putParameter("k", 5.0);
	}

	@Override
	public void train() {
		int S = (int)this.getDoubleParameter("S");
		int k = (int)this.getDoubleParameter("k");
		Map<String, DataSet<E>> trainingSets = this.splitTrainingData(this.trainingDataSet);
		this.randomlyAssignClasses(this.trainingDataSet, S, trainingSets);
		
		for(int iteration = 0; iteration < k; iteration++){
			int changes = 0;
			for(String className : trainingSets.keySet()){
				DataSet<E> data = trainingSets.get(className);
				this.printStatistics(className, data, iteration);
				FileUtil.mkdir(baseDir + className);
				baseClassifier.setBaseDir(baseDir + className + "/");
				baseClassifier.setTrainingSet(data);
				baseClassifier.train();
				baseClassifier.setTestSet(data);
				baseClassifier.test();
				for(Instance each : data){
					if(!each.groundTruth.equals(each.label)) changes++;
					each.groundTruth = each.label;
				}
			}
			Log.puts("Number of subclass changes in iteration " + iteration + ": " + changes);
			if(changes == 0) break;
		}
		
		FileUtil.mkdir(baseDir + "all/");
		baseClassifier.setBaseDir(baseDir + "all/");
		baseClassifier.setTrainingSet(this.trainingDataSet);
		baseClassifier.train();
		this.restoreClasses(this.trainingDataSet);
	}
	
	@Override
	public DataSet<E> test(){
		baseClassifier.setBaseDir(baseDir + "all/");
		baseClassifier.setTestSet(this.testDataSet);
		this.testDataSet = baseClassifier.test();
		for(Instance each : testDataSet)
			each.label = each.label.split("-")[0];
		return this.testDataSet;
	}

	private void printStatistics(String className, DataSet<E> data, int iteration) {
		Log.puts("Iteration " + iteration);
		Log.puts("Class: " + className);
		Map<String, Integer> classDistribution = new LinkedHashMap<String, Integer>();
		for(Instance each : data){
			if(!classDistribution.containsKey(each.groundTruth))
				classDistribution.put(each.groundTruth, 0);
			classDistribution.put(each.groundTruth, classDistribution.get(each.groundTruth) + 1);
		}
		Log.puts("Distribution: " + classDistribution);
	}

	private Map<String, DataSet<E>> splitTrainingData(DataSet<E> data) {
		Map<String, DataSet<E>> sets = new LinkedHashMap<String, DataSet<E>>();
		for(E each : data){
			if(!sets.containsKey(each.groundTruth))
				sets.put(each.groundTruth, new DataSet<E>());
			sets.get(each.groundTruth).addInstance(each);
		}
		return sets;
	}

	private void restoreClasses(DataSet<E> data) {
		for(Instance each : data)
			each.groundTruth = each.groundTruth.split("-")[0];
	}

	private void randomlyAssignClasses(DataSet<E> data, int s, Map<String, DataSet<E>> trainingSets) {
		int maxInstancesPerClass = 0;
		for(DataSet<E> set : trainingSets.values())
			maxInstancesPerClass = Math.max(maxInstancesPerClass, set.size());
		for(Instance each : data){
			int si = (int)(s * trainingSets.get(each.groundTruth).size()/(double)maxInstancesPerClass);
			if(si < 1) si = 1;
			each.groundTruth = each.groundTruth + "-" + ((int)(Math.random()*si));
		}
	}

}

package ch.eonum.pipeline.classification.meta;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Log;

/**
 * Clustered classifier. Creates a classifier for each distinct className in the
 * training set. instances in the test set will be classified with the
 * classifier according their class name. The wrapped classifiers do not have to
 * be serializable/deserializable but the entire map of classifiers for each
 * class name has to be provided. @see ClusteredClassifier for a slightly
 * different implementation using only one wrapped classifier.
 * 
 * @author tim
 * 
 */
public class ClusteredClassifierWithList<E extends Instance> extends Classifier<E> {
	private Set<String> clusters;
	private Map<String, Double> resultAverages;
	private Map<String, Double> ratios;
	private Map<String, Classifier<E>> classifiers;

	public ClusteredClassifierWithList(Map<String, Classifier<E>> classifiers) {
		super();
		this.classifiers = classifiers;
		initClusteredClassifier();
	}
	
	public ClusteredClassifierWithList(){
		super();
		initClusteredClassifier();
	}
	
	private void initClusteredClassifier() {
		this.resultAverages = new HashMap<String, Double>();
		this.ratios = new HashMap<String, Double>();
	}
	
	@Override
	public void train() {
		Map<String, DataSet<E>> clusteredTrainingData = this.clusterData(this.trainingDataSet);
		Map<String, DataSet<E>> clusteredTestData = this.clusterData(this.testDataSet);

		this.clusters = clusteredTrainingData.keySet();
		DataSet<E> oldTrainData = this.trainingDataSet;
		DataSet<E> oldTestData = this.testDataSet;


		for(String cluster : this.clusters){
			int numInst = clusteredTrainingData.get(cluster).size();	
			Log.puts("Training of cluster: " + cluster + " " + numInst + " instances.");
			this.trainingDataSet = clusteredTrainingData.get(cluster);
			this.testDataSet = clusteredTestData.get(cluster);
			double ratio = clusteredTrainingData.get(cluster).getRatio();
			this.ratios.put(cluster, ratio);
			
			Classifier<E> classifier = classifiers.get(cluster);
			this.updateDataSets(classifier);
			classifier.train();
			this.calculateAverageResult(cluster);
		}
		this.trainingDataSet = oldTrainData;
		this.testDataSet = oldTestData;
	}

	private void updateDataSets(Classifier<E> classifier) {
		classifier.setTrainingSet(this.trainingDataSet);
		classifier.setTestSet(this.testDataSet);
	}

	private void calculateAverageResult(String cluster) {
		if(!this.resultAverages.containsKey(cluster)){
			Classifier<E> classifier = classifiers.get(cluster);
			DataSet<E> testSet = this.testDataSet;
			this.testDataSet = this.trainingDataSet;
			this.updateDataSets(classifier);
			classifier.test();
			double sum = 0.0;
			for(Instance each : this.trainingDataSet)
				sum += each.getResult("result");
			this.resultAverages.put(cluster, sum/this.trainingDataSet.size());
			this.testDataSet = testSet;
		}
	}

	@Override
	public DataSet<E> test() {
		Map<String, DataSet<E>> clusteredTestData = this.clusterData(this.testDataSet);
		DataSet<E> testData = this.testDataSet;
		for(String cluster : clusteredTestData.keySet()){
			Log.puts("Testing cluster: " + cluster + " " + clusteredTestData.get(cluster).size() + " instances.");

			this.testDataSet = clusteredTestData.get(cluster);
			Classifier<E> classifier = classifiers.get(cluster);
			this.updateDataSets(classifier);
			classifier.test();
		}
		this.testDataSet = testData;
		return this.testDataSet;
	}
	
	private Map<String, DataSet<E>> clusterData(DataSet<E> data) {
		Map<String, DataSet<E>> clusteredData = new HashMap<String, DataSet<E>>();
		for(E each : data){
			if(!clusteredData.containsKey(each.className))
				clusteredData.put(each.className, new DataSet<E>());
			
			clusteredData.get(each.className).addInstance(each);
		}
		return clusteredData;
	}

}

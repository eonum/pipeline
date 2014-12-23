package ch.eonum.pipeline.classification.meta;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.util.Log;

/**
 * Clustered classifier. Creates a classifier of the wrapped type for each
 * distinct className in the training set. instances in the test set will be
 * classified with the classifier according their class name.
 * 
 * @author tim
 * 
 */
public class ClusteredClassifier<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("instancesThreshold", "minimum number of instances for a cluster. if it's under the threshold, the garbage model is used (default: 1");
	}

	private Set<String> clusters;
	private Map<String, Double> resultAverages;
	private Map<String, Double> ratios;
	private Classifier<E> classifier;
	private Map<String, Features> featuresPerClass;

	public ClusteredClassifier(Classifier<E> classifier) {
		super();
		this.classifier = classifier;
		initClusteredClassifier();
	}
	
	public ClusteredClassifier(){
		super();
		initClusteredClassifier();
	}
	
	private void initClusteredClassifier() {
		this.setSupportedParameters(ClusteredClassifier.PARAMETERS);
		this.putParameter("instancesThreshold", 1.0);
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
		String oldBaseDir = classifier.getBaseDir();
		ArrayList<String> smallClustersToBeRemoved = new ArrayList<String>();
		for(String cluster : this.clusters){
			int numInst = clusteredTrainingData.get(cluster).size();
			if(numInst < this.getDoubleParameter("instancesThreshold")) {
				smallClustersToBeRemoved.add(cluster);
				continue;
			}
			Log.puts("Training of cluster: " + cluster + " " + numInst + " instances.");
			this.trainingDataSet = clusteredTrainingData.get(cluster);
			this.testDataSet = clusteredTestData.get(cluster);
			double ratio = clusteredTrainingData.get(cluster).getRatio();
			this.ratios.put(cluster, ratio);
			classifier.putParameter("w", ratio);
			FileUtil.mkdir(oldBaseDir + cluster + "/");
			classifier.setBaseDir(oldBaseDir + cluster + "/");
			this.updateDataSets();
			if(this.featuresPerClass != null)
				classifier.setFeatures(this.featuresPerClass.get(cluster));
			classifier.train();
			this.calculateAverageResult(cluster);
		}
		for(String cluster : smallClustersToBeRemoved)
			this.clusters.remove(cluster);
		this.trainingDataSet = oldTrainData;
		this.testDataSet = oldTestData;
		classifier.setBaseDir(oldBaseDir);
		
		if(!smallClustersToBeRemoved.isEmpty()){
		Log.puts("Training garbage model");
			double ratio = trainingDataSet.getRatio();
			this.ratios.put("garbage", ratio);
			classifier.putParameter("w", ratio);
			this.updateDataSets();
			classifier.setFeatures(this.features);
			classifier.train();
			this.calculateAverageResult("garbage");
		}
	}

	private void updateDataSets() {
		this.classifier.setTrainingSet(this.trainingDataSet);
		this.classifier.setTestSet(this.testDataSet);
	}

	private void calculateAverageResult(String cluster) {
		if(!this.resultAverages.containsKey(cluster)){
			DataSet<E> testSet = this.testDataSet;
			this.testDataSet = this.trainingDataSet;
			this.updateDataSets();
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
		String oldBaseDir = classifier.getBaseDir();
		for(String cluster : clusteredTestData.keySet()){
			Log.puts("Testing cluster: " + cluster + " " + clusteredTestData.get(cluster).size() + " instances.");

			this.testDataSet = clusteredTestData.get(cluster);
			if(!this.clusters.contains(cluster)){
				/** use garbage model. */
				classifier.setBaseDir(oldBaseDir);
			} else {
				classifier.setBaseDir(oldBaseDir + cluster + "/");
			}
			if(this.featuresPerClass != null)
				classifier.setFeatures(this.featuresPerClass.get(cluster));
			this.updateDataSets();
			classifier.test();
			/** normalize. */
			for(Instance each : this.testDataSet){
				if(!this.clusters.contains(cluster))
					each.putResult("result", each.getResult("result"));
				else
					each.putResult("result", each.getResult("result"));
			}
		}
		this.testDataSet = testData;
		classifier.setBaseDir(oldBaseDir);
		classifier.setFeatures(features);
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

	public void setFeaturesPerClass(Map<String, Features> dimsPerClass) {
		this.featuresPerClass = dimsPerClass;
	}

}

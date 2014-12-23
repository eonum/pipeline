package ch.eonum.pipeline.classification;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.core.DataPipeline;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Parameters;
import ch.eonum.pipeline.util.json.JSON;


/**
 * <p>Classifier or Regressor</p>
 * Train using a training data set and classify/test a given test data set.
 * 
 * @author tim
 *
 */
public abstract class Classifier<E extends Instance> extends Parameters implements DataPipeline<E>, Runnable {
	/** test data set. used either for validation or for testing/productive use. */
	protected DataSet<E> testDataSet;
	/** training data set. used for creating the classifier or regressor. */
	protected DataSet<E> trainingDataSet;
	/** base directory to print files. */
	protected String baseDir;
	/** preceding pipeline object in the data flow training data. */
	private DataPipeline<E> inputTraining;
	/** preceding pipeline object in the data flow test data. */
	private DataPipeline<E> inputTest;
	/** input features used by this classifier. */
	protected Features features;
	/**
	 * classes associated with an index. output features. null if this is a
	 * regression task.
	 */
	protected Features classes;
	/**
	 * is this a classification problem. default is false.
	 */
	protected boolean classify;

	/**
	 * every Classifier must have a zero argument constructor
	 */
	public Classifier(){}
	
	public void setFeatures(Features features){
		this.features = features;
	}
	
	public Features getFeatures(){
		return this.features;
	}
	
	public void setTrainingSet(DataSet<E> dataset){
		this.trainingDataSet = dataset;
	}
	
	public void setTestSet(DataSet<E> dataset){
		this.testDataSet = dataset;
	}
	
	/**
	 * train the classifier using the training set.
	 */
	public abstract void train();
	
	/**
	 * Make sure the classifier is loaded before calling this method. classify
	 * the test set. get the classified data set with a probability measure in
	 * the result "result". classification with probability measures: All
	 * probabilities/likelihoods are stored in the result variables
	 * "classProb<class>"
	 * 
	 * @throws IOException
	 */
	public DataSet<E> test() {
		return this.testDataSet;
	}
	
	/**
	 * Test one instance. test the instance once for each feature, which is set
	 * to zero. the feature is stored in the label of the created instance.
	 * "base" is an additional instance where nothing has been removed.
	 * 
	 * 
	 * @param inst
	 * @return a data set with an instance for each feature of this instance and
	 *         their relative change compared to the base case (no features
	 *         removed).
	 */
	@SuppressWarnings("unchecked")
	public DataSet<E> testVariants(E inst) {
		DataSet<E> temp = this.testDataSet;
		this.testDataSet = new DataSet<E>();
		E i = (E)inst.copy();
		i.id = "base";
		this.testDataSet.add(i);
		for(String feature : inst.features()){
			i = (E)inst.copy();
			i.remove(feature);
			i.id = feature;
			this.testDataSet.add(i);
		}
		DataSet<E> res = test();
		this.testDataSet = temp;
		return res;
	}
	
	/**
	 * @see #baseDir
	 * @param baseDir
	 */
	public void setBaseDir(String baseDir) {
		this.baseDir = baseDir;
	}

	/**
	 * @see #baseDir
	 * @return
	 */
	public String getBaseDir() {
		return baseDir;
	}
	
	/**
	 * serialize the current state of this classifier.
	 * child classes might have to extend this method.
	 * 
	 * @param fileName
	 * @throws IOException 
	 */
	public void save(String fileName) throws IOException {
		Map<String, Object> dbo = asMap();
		JSON.writeJSON(new File(fileName), dbo);
	}
	
	@Override
	public Map<String, Object> asMap(){
		Map<String, Object> dbo = super.asMap();
		dbo.put("classify", classify);
		List<String> list;
		if(classes != null){
			list = classes.asStringList();
			dbo.put("classes", list);
		}
		if(features != null){
			list = features.asStringList();
			dbo.put("features", list);
		}
		dbo.put("classify", classify);
		dbo.put("baseDir", baseDir);
		return dbo;
	}
	
	/**
	 * load serialized state of this classifier.
	 * subclasses might extend this method.
	 * @param file
	 * @throws IOException 
	 */
	public void loadSerializedState(File file) throws IOException {
		Map<String, Object> data = JSON.readJSON(file);
		super.load(data);
		this.load(data);
	}
	
	@SuppressWarnings("unchecked")
	@Override
	protected void load(Map<String, Object> data) {
		this.classify = (Boolean) data.get("classify");
		this.baseDir = (String) data.get("baseDir");
		List<String> list;
		if(data.containsKey("classes")){
			list = (List<String>) data.get("classes");
			this.classes = new Features(list);
		}
		if(data.containsKey("features")){
			list = (List<String>) data.get("features");
			this.features = new Features(list);
		}
	}
	
	@Override
	public void run(){
		this.train();
	}
	
	public void prepareClasses() {
		this.classes = new Features();
		for(String className : trainingDataSet.collectClasses())
			classes.addFeature(className);
		for(String className : testDataSet.collectClasses())
			classes.addFeature(className);
		classes.recalculateIndex();
	}
	
	/**
	 * Represents this classifier a classification a classification or a
	 * regression task.
	 * 
	 * @return
	 */
	public boolean isClassifier(){
		return classify;
	}

	/** pipeline methods . **/
	
	@Override
	public DataSet<E> trainSystem(boolean isResultDataSetNeeded){
		if(this.inputTraining != null)
			trainingDataSet = this.inputTraining.trainSystem(true);
		if(this.inputTest != null)
			this.testDataSet = this.inputTest.testSystem();
		train();
		if(isResultDataSetNeeded)	{
			DataSet<E> temp = this.testDataSet;
			this.testDataSet = this.trainingDataSet;
			test();
			DataSet<E> temp2 = this.testDataSet;
			this.testDataSet = temp;
			return temp2;
		}
		return null;
	}
	
	@Override
	public DataSet<E> testSystem(){
		if(this.inputTest != null)
			this.testDataSet = this.inputTest.testSystem();
		return test();
	}
	
	@Override
	public void addInputTraining(DataPipeline<E> input){
		this.inputTraining = input;
	}
	
	@Override
	public void addInputTest(DataPipeline<E> input){
		this.inputTest = input;
	}

	public DataSet<E> getTestDataSet() {
		return this.testDataSet;
	}

	public DataSet<E> getTrainingDataSet() {
		return this.trainingDataSet;
	}
	
	public Features getClasses(){
		return classes;
	}
	
}

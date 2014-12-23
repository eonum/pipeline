package ch.eonum.pipeline.core;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * One vector/sample/instance/record/entry. An instance can be either densely
 * represented (@see DenseInstance) or sparsely (@see SparseInstance).
 * 
 * @author tim
 * 
 */
public abstract class Instance {

	/** identification of this item. */
	public String id;
	/** ground truth. class label. */
	public String groundTruth;
	/** groundTruth for regression tasks. */
	public double outcome;
	/** weight for this instance default: 1.0. */
	public double weight;
	/** class/cluster of this instance. Used for clustered classifiers. */
	public String className;
	/** predicted class label after classification. */
	public String label;
	/**
	 * predicted values for regression / prediction likelihoods and class
	 * probabilities for classification.
	 */
	public Map<String, Double> results;
	
	public Instance() {
		super();
	}

	public Instance(String id, String gt) {
		this.id = id;
		this.groundTruth = gt;
		this.weight = 1.0;
	}

	/**
	 * exchange a certain feature with another instance. Used for randomization
	 * (permutation) of certain features.
	 * 
	 * @param feature
	 * @param instance
	 */
	public void exchangeFeature(String feature, Instance instance) {
		double value = get(feature);
		double otherValue = instance.get(feature);
		put(feature, otherValue);
		instance.put(feature, value);
	}

	public void addAll(Instance member) {
		for(String f : member.features())
			this.put(f, member.get(f));
	}

	public abstract boolean hasFeature(String feature);

	/**
	 * Remove all features in this instance, which are not in the provided
	 * feature set
	 * 
	 * @param feature
	 *            set
	 */
	public abstract void reduceFeatures(Features features);

	/**
	 * Set a new data map. All old values are discarded. The pointer to the old
	 * data is being kept, which means that the map/vector itself is not deleted.
	 * 
	 * @param v
	 *            new data
	 */
	public abstract void put(Map<String, Double> v);

	public Map<String, Object> asTree() {
		LinkedHashMap<String, Object> map = new LinkedHashMap<String, Object>();
		map.put("id", this.id);
		if (this.className != null)
			map.put("className", this.className);
		if (this.label != null)
			map.put("label", this.label);
		map.put("groundTruth", this.groundTruth);
		return map;
	}

	public abstract String toString();

	public abstract void cleanUp();

	/**
	 * take the square root of each value. the instance is changed.
	 */
	public void sqrt() {
		for (String feature : this.features())
			put(feature, Math.sqrt(get(feature)));
	}

	/**
	 * take the natural log of each value + 1. the instance is changed.
	 */
	public void log1p() {
		for (String feature : this.features())
			put(feature, Math.log1p(get(feature)));
	}

	/**
	 * take the power of each value. the instance is changed.
	 */
	public void pow(double power) {
		for (String feature : this.features())
			put(feature, Math.pow(get(feature), power));
	}

	public abstract Instance copy();

	/**
	 * multiply each feature. the instance is not changed. a newly created
	 * instance is returned. inst2.times(inst1) == inst1.times(inst2) But for
	 * better performance you should first put the instance with more features.
	 * 
	 * @param each
	 * @return
	 */
	public Instance timesStateless(Instance i) {
		SparseInstance inst = new SparseInstance(id, groundTruth,
				new HashMap<String, Double>());
		inst.className = className;
		inst.outcome = outcome;
		for (String feature : i.features())
			inst.put(feature, i.get(feature) * get(feature));
		inst.cleanUp();
		return inst;
	}

	/**
	 * multiply each feature. the instance is changed.
	 * 
	 * @param each
	 * @return
	 */
	public void times(Instance i) {
		for (String feature : i.features())
			this.put(feature, i.get(feature) * get(feature));
		this.cleanUp();
	}

	/**
	 * multiply each feature with a factor. the instance is not changed. a newly
	 * created instance is returned.
	 * 
	 * @param factor
	 * @return
	 */
	public SparseInstance times(double factor) {
		SparseInstance inst = new SparseInstance(id, groundTruth,
				new HashMap<String, Double>());
		inst.className = className;
		for (String feature : features())
			inst.put(feature, get(feature) * factor);
		inst.cleanUp();
		return inst;
	}

	/**
	 * subtract instance i from this instance and return a new instance with the
	 * result. the instance is not changed.
	 * 
	 * @param i
	 * @return
	 */
	public Instance minusStateless(Instance i) {
		Instance inst = this.copy();
		for (String feature : i.features())
			inst.put(feature, inst.get(feature) - i.get(feature));
		return inst;
	}

	/**
	 * subtract instance i from this instance. the instance is changed.
	 * 
	 * @param mean
	 */
	public void minus(Instance i) {
		for (String feature : i.features())
			put(feature, get(feature) - i.get(feature));
	}
	
	/**
	 * add an instance to this instance. the instance is changed.
	 * 
	 * @param inst
	 */
	public void add(Instance inst) {
		for (String feature : inst.features())
			put(feature, inst.get(feature) + get(feature));
	}

	/**
	 * divide all features by divisor. the instance is changed.
	 * 
	 * @param divisor
	 */
	public void divideBy(double divisor) {
		for (String feature : features())
			put(feature, get(feature) / divisor);
	}

	/**
	 * divide by another instance. only features within divisor are being
	 * divided. the instance is changed.
	 * 
	 * @param divisor
	 */
	public void divideBy(Instance divisor) {
		for (String feature : divisor.features())
			if (hasFeature(feature)) // avoid 0-entries
				put(feature, get(feature) / divisor.get(feature));
	}

	public abstract double remove(String feature);

	public abstract Set<String> features();

	public abstract void put(String feature, double value);

	public abstract double get(String feature);
	
	public abstract double[] asArray(Features features);
		
	/**
	 * store a prediction result.
	 * @param key
	 * @param value
	 */
	public void putResult(String key, double value){
		if(this.results == null)
			this.results = new HashMap<String, Double>();
		results.put(key, value);
	}
	
	/**
	 * get a prediction result / probability estimation / prediction accuracy. 
	 * Depends on classification / regression model being used.
	 * 
	 * @param key
	 * @return
	 */
	public double getResult(String key){
		return (results == null || !results.containsKey(key)) ? 0.0 : results.get(key);
	}

	public void removeResult(String key) {
		this.results.remove(key);
	}

}
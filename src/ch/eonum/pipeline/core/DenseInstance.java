/**
 * 
 */
package ch.eonum.pipeline.core;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

/**
 * Dense representation of an @see Instance using a simple double vector.
 * @author tim
 *
 */
public class DenseInstance extends Instance {

	/** features for the mapping of vector indices to features. */
	private Features features;
	/** dense data. */
	private double[] vector;

	/**
	 * constructor.
	 */
	public DenseInstance(String id, String gt, Features features) {
		super(id, gt);
		this.features = features;
		this.vector = new double[features.size()];
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#hasFeature(java.lang.String)
	 */
	@Override
	public boolean hasFeature(String feature) {
		return features.hasFeature(feature);
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#reduceFeatures(ch.eonum.pipeline.core.Features)
	 * 
	 * set the features pointer of this instance to the provided feature set.
	 */
	@Override
	public void reduceFeatures(Features features) {
		double[] newVector = new double[features.size()];
		for(int i = 0; i < features.size(); i++){
			String feature = features.getFeatureByIndex(i);
			newVector[i] = this.features.hasFeature(feature) ? vector[this.features.getIndexFromFeature(feature)] : 0.0;
		}
		vector = newVector;
		this.features = features;
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#put(java.util.Map)
	 */
	@Override
	public void put(Map<String, Double> v) {
		for(int i = 0; i < vector.length; i++){
			String feature = features.getFeatureByIndex(i);
			vector[i] = v.containsKey(feature) ? v.get(feature) : 0.0;
		}	
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#asTree()
	 */
	@Override
	public Map<String, Object> asTree() {
		Map<String, Object> map = super.asTree();
		map.put("vector", Arrays.asList(this.vector));
		return map;
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#toString()
	 */
	@Override
	public String toString() {
		String r = "[Id: " + id + " GroundTruth: " + groundTruth;
		r += " label: " + (label == null ? "" : label);
		return r + " Features: " + Arrays.asList(vector) + "]";
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#cleanUp()
	 */
	@Override
	public void cleanUp() {
		// do nothing
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#copy()
	 */
	@Override
	public DenseInstance copy() {
		DenseInstance inst = new DenseInstance(id, groundTruth, features);
		inst.outcome = outcome;
		inst.weight = weight;
		inst.className = className;
		for(int i = 0; i < vector.length; i++)
			inst.vector[i] = vector[i];
		return inst;
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#remove(java.lang.String)
	 */
	@Override
	public double remove(String feature) {
		double value = get(feature);
		put(feature, 0.0);
		return value;
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#features()
	 */
	@Override
	public Set<String> features() {
		return features.asSet();
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#put(java.lang.String, double)
	 */
	@Override
	public void put(String feature, double value) {
		vector[features.getIndexFromFeature(feature)] = value;
	}

	/**
	 * @see ch.eonum.pipeline.core.Instance#get(java.lang.String)
	 */
	@Override
	public double get(String feature) {
		return vector[features.getIndexFromFeature(feature)];
	}

	@Override
	public double[] asArray(Features features) {
		assert(features == this.features);
		return vector;
	}

}

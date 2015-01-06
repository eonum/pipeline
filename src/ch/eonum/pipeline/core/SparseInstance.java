package ch.eonum.pipeline.core;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ch.eonum.pipeline.util.Log;
import ch.eonum.pipeline.util.json.JSONable;

/**
 * Sparse implementation of an {@link Instance} using a map.
 * 
 * @see Dataset
 * 
 * @author tim
 * 
 */
public class SparseInstance extends Instance implements JSONable, Cloneable {
	private static Pattern pattern;

	/** sparse vector holding all features unlike zero of this instance. */
	protected Map<String, Double> vector;
	/**
	 * Constructor
	 * 
	 * @param id
	 * @param gt
	 *            GroundTruth class label
	 * @param vector
	 *            data describing this instance
	 */
	public SparseInstance(String id, String gt, Map<String, Double> vector) {
		super(id, gt);
		this.vector = vector;
	}

	/**
	 * Constructor
	 * 
	 * @param id
	 * @param groundTruth
	 * @param newVector
	 * @param className
	 */
	public SparseInstance(String id, String groundTruth, Map<String, Double> vector,
			String className) {
		this(id, groundTruth, vector);
		this.className = className;
	}

	/**
	 * get the value of a feature. if there is no such feature, 0 is returned.
	 * 
	 * @param feature
	 * @return the value of the given feature.
	 */
	@Override
	public double get(String feature) {
		return vector.containsKey(feature) ? vector.get(feature) : 0.0;
	}

	/**
	 * store a value for a certain feature
	 * 
	 * @param feature
	 * @param value
	 */
	@Override
	public void put(String feature, double value) {
		vector.put(feature, value);
	}

	@Override
	public Set<String> features() {
		return vector.keySet();
	}

	@Override
	public double remove(String feature) {
		return vector.containsKey(feature) ? this.vector.remove(feature) : 0.0;
	}
	
	@Override
	public SparseInstance copy() {
		Map<String, Double> newVector = new HashMap<String, Double>(vector);
		SparseInstance inst = new SparseInstance(id, groundTruth, newVector, className);
		inst.outcome = outcome;
		inst.weight = weight;
		return inst;
	}

	/**
	 * remove all 0 entries for better memory usage or to avoid division by zero
	 * errors.
	 */
	@Override
	public void cleanUp() {
		String[] dimensions = features().toArray(new String[0]);
		for (String feature : dimensions)
			if (get(feature) > -0.000000001 && get(feature) < 0.000000001)
				vector.remove(feature);
	}

	@Override
	public String toString() {
		String r = "[Id: " + id + " GroundTruth: " + groundTruth;
		r += " label: " + (label == null ? "" : label);
		return r + " Features: " + vector + "]";
	}

	/**
	 * parse a line and create an instance. the line is formatted in the format
	 * specified in {@link #toString()}
	 * 
	 * @param line
	 * @return
	 */
	public static SparseInstance parse(String line) {
		/** lazy initialization. */
		if (pattern == null)
			pattern = Pattern
					.compile("\\[Id: ([^\\s]*) GroundTruth: ([^\\s]*) label: ([^\\\\s]*) Features: \\{([^\\}]*)\\}\\]");
		Matcher matcher = pattern.matcher(line);
		Map<String, Double> vector = new HashMap<String, Double>();
		if (matcher.find()) {
			for (String feature : matcher.group(4).split(",")) {
				String[] split = feature.split("=");
				vector.put(split[0].trim(), Double.valueOf(split[1].trim()));
			}
		} else {
			Log.warn("Wrong Format for line: " + line);
		}
		return new SparseInstance(matcher.group(1), matcher.group(2), vector,
				matcher.group(3));
	}

	@Override
	public Map<String, Object> asTree() {
		Map<String, Object> map = super.asTree();
		map.put("vector", this.vector);
		return map;
	}

	@SuppressWarnings("unchecked")
	@Override
	public void readFromTree(Map<String, Object> tree)
			throws InstantiationException, IllegalAccessException,
			ClassNotFoundException {
		this.id = (String) tree.get("id");
		if (tree.containsKey("className"))
			this.className = (String) tree.get("className");
		if (tree.containsKey("label"))
			this.label = (String) tree.get("label");
		this.groundTruth = (String) tree.get("groundTruth");
		this.vector = (Map<String, Double>) tree.get("vector");
	}

	@Override
	public void put(Map<String, Double> v) {
		this.vector.clear();
		for (String each : v.keySet())
			this.put(each, v.get(each));
	}

	@Override
	public void reduceFeatures(Features features) {
		Set<String> set = new HashSet<String>();
		for (String feature : this.features())
			set.add(feature);
		for (String feature : set)
			if (!features.hasFeature(feature))
				this.remove(feature);
	}

	@Override
	public boolean hasFeature(String feature) {
		return this.vector.containsKey(feature);
	}

	@Override
	public double[] asArray(Features features) {
		double[] values = new double[features.size()];
		for (int f = 0; f < features.size(); f++)
			values[f] = get(features.getFeatureByIndex(f));
		return values;
	}
	
	/**
	 * Get an {@link Entry} array with the provided features from this instance.
	 * @param features
	 * @return
	 */
	public Entry[] asEntryArray(Features features) {
		Set<String> set = new HashSet<String>();
		for (String feature : this.features())
			if(features.hasFeature(feature))
				set.add(feature);
		Entry[] values = new Entry[set.size()];
		int f = 0;
		for (String feature : set)
			values[f++] = new Entry(features.getIndexFromFeature(feature), get(feature));
		return values;
	}

}

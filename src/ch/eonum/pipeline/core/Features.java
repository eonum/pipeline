package ch.eonum.pipeline.core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ch.eonum.pipeline.util.Log;

/**
 * A set holding with dimensions/features.
 * 
 * Each feature is identified by a unique integer and is associated to an index.
 * 
 * The set can be read or written to a file. Feature labels are ordered and can
 * be accessed by an index
 * 
 * @author tim
 * 
 */
public class Features {
	private List<String> featuresByIndex;
	private Map<String, Integer> indicesByFeature;
	private Map<String, String> descriptions;

	/**
	 * Constructor. Create empty feature set.
	 */
	public Features() {
		this.featuresByIndex = new ArrayList<String>();
		this.indicesByFeature = new LinkedHashMap<String, Integer>();
		this.descriptions = new HashMap<String, String>();
	}

	/**
	 * Constructor. Initialize using an existing list of features.
	 * @param list
	 */
	public Features(List<String> list) {
		this();
		for(String e : list)
			this.addFeature(e);
		this.recalculateIndex();
	}

	/**
	 * get the index from a certain feature.
	 * 
	 * @param feature
	 * @return
	 */
	public int getIndexFromFeature(String feature) {
		return this.indicesByFeature.get(feature);
	}

	/**
	 * get the feature label from a certain index.
	 * 
	 * @param index
	 * @return
	 */
	public String getFeatureByIndex(int index) {
		return this.featuresByIndex.get(index);
	}

	/**
	 * Get the description of a feature. Return the feature itself if there is
	 * no description for this feature.
	 * 
	 * @param dim
	 * @return
	 */
	public String getDescription(String feature) {
		return this.descriptions.containsKey(feature) ? this.descriptions
				.get(feature) : feature;
	}

	/**
	 * Print all features to a file.
	 * 
	 * @param fileName
	 */
	public void writeToFile(String fileName) {
		try {
			FileWriter fstream = new FileWriter(fileName);
			BufferedWriter out = new BufferedWriter(fstream);
			for (int i = 0; i < this.featuresByIndex.size(); i++)
				out.write(i + ":" + featuresByIndex.get(i) + ":"
						+ this.getDescription(featuresByIndex.get(i)) + "\n");

			out.close();
		} catch (Exception e) {
			Log.warn(e.getMessage());
		}
	}

	/**
	 * Read a features file into memory. All lines of a features file are in the
	 * format: "index:feature_label:description" Example:
	 * "23:los:Length of stay" The indices are not read. They are asserted
	 * to be equal to the line number -1. The only purpose of the indices in the
	 * features file is better readability.
	 * 
	 * @param fileName
	 * @return
	 */
	public static Features readFromFile(String fileName) {
		Features dimensions = new Features();

		FileInputStream fstream;
		try {
			fstream = new FileInputStream(fileName);

			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String line;

			while ((line = br.readLine()) != null) {
				String[] split = line.split(":");
				dimensions.addFeature(split[1]);
				if (split.length > 2)
					dimensions.addDescription(split[1], split[2]);
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		return dimensions;
	}

	/**
	 * Add a description to a feature
	 * 
	 * @param feature
	 * @param description
	 */
	private void addDescription(String feature, String description) {
		this.descriptions.put(feature, description);
	}

	/**
	 * Create a feature object from a list of strings, usually extracted
	 * using a FeatureExtractor
	 * 
	 * @param fileName
	 * @return
	 */
	public static Features createFromList(List<String> feats) {
		return new Features(feats);
	}

	/**
	 * Get the number of features.
	 * 
	 * @return
	 */
	public int size() {
		return this.featuresByIndex.size();
	}

	/**
	 * Load descriptions of features from a separate file. format (one line):
	 * feature:description
	 * 
	 * @param fileName
	 *            file name of the descriptions file
	 */
	public void loadDescriptionFile(String fileName) {
		FileInputStream fstream;
		try {
			fstream = new FileInputStream(fileName);

			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String line;

			while ((line = br.readLine()) != null) {
				String[] split = line.split(":");
				if (split.length >= 2) {
					String description = "";
					for (int i = 1; i < split.length; i++)
						description += split[i];
					this.addDescription(split[0], description);
				} else
					Log.warn("Unexpected description format in line: " + line);
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	/**
	 * Remove a feature. The index is not recalculated. You have to do this
	 * separately by calling {@link #recalculateIndex()}
	 * 
	 * @param feature
	 */
	public void removeFeature(String feature) {
		this.indicesByFeature.remove(feature);
	}

	/**
	 * Add a feature. If you try to add an existing feature, this method has no
	 * effect.
	 * 
	 * @param feature
	 */
	public void addFeature(String feature) {
		if (!this.featuresByIndex.contains(feature)) {
			this.featuresByIndex.add(feature);
			this.indicesByFeature.put(feature, this.featuresByIndex.size() - 1);
		}
	}

	/**
	 * Recalculate the index. this is typically done after removing some
	 * features.
	 */
	public void recalculateIndex() {
		this.featuresByIndex = new ArrayList<String>(
				this.indicesByFeature.keySet());
		this.indicesByFeature = new HashMap<String, Integer>();
		for (int i = 0; i < this.featuresByIndex.size(); i++)
			this.indicesByFeature.put(this.featuresByIndex.get(i), i);
	}

	/**
	 * Create a data set with an instance for each feature. An instance has only
	 * one feature with the value 1.0
	 * 
	 * @return
	 */
	public DataSet<SparseInstance> createDataSet() {
		DataSet<SparseInstance> ds = new DataSet<SparseInstance>();
		for (String feature : this.featuresByIndex) {
			HashMap<String, Double> data = new HashMap<String, Double>();
			data.put(feature, 1.0);
			ds.add(new SparseInstance(feature, feature, data));
		}
		return ds;
	}

	/**
	 * Check if a feature exists in this feature set.
	 * @param feature
	 * @return
	 */
	public boolean hasFeature(String feature) {
		return this.featuresByIndex.contains(feature);
	}

	/**
	 * Get a copy of the list of all feature identifiers.
	 * @return
	 */
	public List<String> getListOfFeaturesCopy() {
		return new ArrayList<String>(this.featuresByIndex);
	}

	@Override
	public String toString() {
		String ret = "";
		for (int i = 0; i < this.featuresByIndex.size(); i++)
			ret += i + ":" + featuresByIndex.get(i) + ":"
					+ this.getDescription(featuresByIndex.get(i)) + "\n";
		return ret;
	}

	/**
	 * Create a feature set that has all features that exist in one of the
	 * provided data sets.
	 * 
	 * @param dataSets
	 * @return
	 */
	@SafeVarargs
	public static Features createFromDataSets(DataSet<? extends Instance> ... dataSets) {
		Features features = new Features();
		for (DataSet<? extends Instance> ds : dataSets)
			for (Instance each : ds)
				for (String feature : each.features())
					features.addFeature(feature);
		features.recalculateIndex();
		return features;
	}

	/**
	 * Create a deep copy of this feature list.
	 * @return
	 */
	public Features copy() {
		Features newF = new Features();
		for (String feature : this.featuresByIndex)
			newF.addFeature(feature);
		newF.recalculateIndex();
		return newF;
	}

	/**
	 * Create a feature set for each class in the provided sparse data sets.
	 * @param dataSets
	 * @return
	 */
	@SafeVarargs
	public static Map<String, Features> createFromDataSetsPerClass(DataSet<? extends SparseInstance> ... dataSets) {
		Map<String, Features> features = new HashMap<String, Features>();
		for (DataSet<? extends SparseInstance> ds : dataSets)
			for (SparseInstance each : ds) {
				if (!features.containsKey(each.className))
					features.put(each.className, new Features());
				for (String feature : each.features())
					features.get(each.className).addFeature(feature);
			}
		for (Features f : features.values())
			f.recalculateIndex();
		return features;
	}

	/**
	 * Remove all features that contain only constant values or are highly
	 * correlated with another feature. Use this to avoid the problem of
	 * multicollinearity in linear regression problems (there is no inverse
	 * matrix if two or more features are perfectly correlated or one or more
	 * features is constant.)
	 * 
	 * @param features
	 * @param dataSet
	 * @return
	 */
	public static Features removeConstantAndPerfectlyCorrellatedFeatures(
			Features features, DataSet<? extends Instance> dataSet) {
		Features newFeatures = new Features();
		double[][] data = dataSet.asDoubleArrayMatrix(features);
		Set<String> constantFeatures = new HashSet<String>();
		for (int f = 0; f < features.size(); f++) {
			double value = data[0][f];
			boolean constant = true;
			for (double[] each : data)
				if (Math.abs(value - each[f]) > 0.00001) {
					constant = false;
					break;
				}
			if (constant) {
				String feature = features.getFeatureByIndex(f);
				Log.warn("Constant feature " + feature + " will be removed.");
				constantFeatures.add(feature);
			}
		}
		double[] means = new double[features.size()];
		for (int f = 0; f < features.size(); f++) {
			double sum = 0.0;
			for (double[] each : data)
				sum += each[f];
			means[f] = sum / data.length;
		}

		for (int f = 0; f < features.size(); f++) {
			String feature = features.getFeatureByIndex(f);
			boolean correlates = false;
			for (int f2a = 0; f2a < newFeatures.size(); f2a++) {
				String feature2 = newFeatures.getFeatureByIndex(f2a);
				int f2 = features.getIndexFromFeature(feature2);
				double cov = 0.0;
				double sd1 = 0.0;
				double sd2 = 0.0;
				for (double[] each : data) {
					double diff1 = each[f] - means[f];
					double diff2 = each[f2] - means[f2];
					cov += diff1 * diff2;
					sd1 += diff1 * diff1;
					sd2 += diff2 * diff2;
				}
				cov /= data.length;
				sd1 /= data.length;
				sd2 /= data.length;

				double corr = cov / (Math.sqrt(sd1) * Math.sqrt(sd2));

				if (Math.abs(corr) > 0.99) {
					Log.warn("Correlated features " + feature + " and "
							+ feature2 + " " + corr + "\n" + feature
							+ " will be removed.");
					correlates = true;
					break;
				}
			}
			if (!correlates) {
				newFeatures.addFeature(feature);
				newFeatures.recalculateIndex();
			}
		}
		for (String f : constantFeatures)
			newFeatures.removeFeature(f);
		newFeatures.recalculateIndex();

		return newFeatures;
	}

	/**
	 * Get all feature identifiers as a set of strings.
	 * @return
	 */
	public Set<String> asSet() {
		return this.indicesByFeature.keySet();
	}

	/**
	 * Get all feature identifiers as a list of strings.
	 * @return
	 */
	public List<String> asStringList() {
		List<String> list = new ArrayList<String>();
		for(int i = 0; i < this.size(); i++)
			list.add(this.getFeatureByIndex(i));
		return list;
	}

}

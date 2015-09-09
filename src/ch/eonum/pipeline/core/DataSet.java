package ch.eonum.pipeline.core;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import ch.eonum.pipeline.analysis.ClassProbability;
import Jama.Matrix;

/**
 * A data set is a vector of instances.
 * 
 * @author tim
 * 
 */
public class DataSet<E extends Instance> extends Vector<E> implements List<E> {

	private static final long serialVersionUID = -5639580196102995883L;

	/**
	 * Empty constructor.
	 */
	public DataSet() {}

	/**
	 * Add all instances of another data set or collection of instances to this
	 * data set.
	 * 
	 * @param values
	 */
	public void addData(Collection<E> values) {
		this.addAll(values);
	}

	/**
	 * Add an instance to this data set.
	 * 
	 * @param instance
	 */
	public void addInstance(E instance) {
		this.add(instance);
	}

	/**
	 * Remove a certain feature from each instance
	 * 
	 * @param feature
	 */
	public void removeFeature(String feature) {
		for (Instance each : this)
			each.remove(feature);
	}

	/**
	 * Print all instances to a file.
	 * 
	 * @param fileName
	 * @throws IOException 
	 */
	public void writeToFile(String fileName) throws IOException {
		PrintStream p = new PrintStream(new FileOutputStream(fileName));
		for (Instance each : this)
			p.println(each);
		p.close();
	}

	/**
	 * Get a deep copy of this data set. Every instance is deep copied.
	 * 
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public DataSet<E> deepCopy() {
		DataSet<E> newSet = new DataSet<E>();
		for (Instance each : this)
			newSet.add((E) each.copy()); // #TODO implement without cast
		return newSet;
	}

	/**
	 * For 2 class problems ("1" and "0") only. return the ratio of the number
	 * of zeros and the number of ones.
	 * 
	 * @return
	 */
	public double getRatio() {
		double one = 1.0;
		double zero = 1.0;
		for (Instance each : this) {
			if ("1".equals(each.groundTruth))
				one++;
			else
				zero++;
		}
		return zero / one;
	}

	/**
	 * Union of all features of all instances in this data set.
	 * 
	 * @return
	 */
	public Set<String> features() {
		HashSet<String> set = new HashSet<String>();
		for (E each : this)
			set.addAll(each.features());
		return set;
	}

	/**
	 * Shuffle the data set. The data set is changed.
	 * Shuffling is deterministic.
	 */
	public void shuffle() {
		this.shuffle(44);
	}
	
	/**
	 * Shuffle the data set. The data set is changed.
	 * Use your own seed.
	 */
	public void shuffle(int seed) {
		Collections.shuffle(this, new Random(seed));
	}

	/**
	 * Get the minimum of a certain feature
	 * 
	 * @param feature
	 * @return minimum
	 */
	public double min(String feature) {
		double min = Double.POSITIVE_INFINITY;
		for (Instance each : this)
			if (each.get(feature) < min)
				min = each.get(feature);
		return min;
	}

	/**
	 * Get the maximum of a certain feature
	 * 
	 * @param feature
	 * @return maximum
	 */
	public double max(String feature) {
		double max = Double.NEGATIVE_INFINITY;
		for (Instance each : this)
			if (each.get(feature) > max)
				max = each.get(feature);
		return max;
	}

	/**
	 * Create a data structure readable for a JSON serializer.
	 * @return
	 */
	public Map<String, Object> asTree() {
		LinkedHashMap<String, Object> map = new LinkedHashMap<String, Object>();
		List<Map<String, Object>> list = new ArrayList<Map<String, Object>>();
		for (E each : this)
			list.add(each.asTree());
		map.put("instances", list);
		return map;
	}

	/**
	 * extract a sub set (e.g. validation set). All instances which are
	 * extracted for the sub set will be randomly removed from this set.
	 * 
	 * The data set is changed.
	 * 
	 * @param p
	 *            percentage of elements which will be used for the sub set.
	 *            range = [0.0,1.0]
	 * @return
	 */
	public DataSet<E> extractSubSet(double p) {
		DataSet<E> ds = new DataSet<E>();
		double i = 0;
		double j = 0;
		for (E each : this) {
			i++;
			if (j / i < p) {
				j++;
				ds.addInstance(each);
			}
		}
		for (Instance each : ds)
			this.remove(each);
		return ds;
	}
	
	/**
	 * Remove randomly selected elements with probability d (0,1].
	 * @param d
	 */
	public void shrink(double d, int seed) {
		Random r = new Random(seed);
		List<E> remove = new ArrayList<E>();
		for (E each : this)
			if(r.nextDouble() < d)
				remove.add(each);
		for(Instance e : remove)
			remove(e);
	}

	/**
	 * Get this data set as a double[][] matrix. Each row represents an
	 * instance. Each column represents a feature.
	 * 
	 * @param features
	 * @return
	 */
	public double[][] asDoubleArrayMatrix(Features features) {
		double[][] matrix = new double[this.size()][];
		for (int i = 0; i < this.size(); i++)
			matrix[i] = get(i).asArray(features);

		return matrix;
	}

	/**
	 * Get this data set as a Jama matrix. Each row represents an instance. Each
	 * column represents a feature.
	 * 
	 * @param features
	 * @return
	 */
	public Matrix asMatrix(Features features) {
		return new Matrix(this.asDoubleArrayMatrix(features));
	}

	/**
	 * Get a set of all classes in this data set.
	 * 
	 * @return
	 */
	public Set<String> collectClasses() {
		Set<String> classes = new HashSet<String>();
		for (Instance e : this)
			classes.add(e.groundTruth);
		return classes;
	}


	/**
	 * Remove all features in all instances, which are not in the provided
	 * feature set
	 * 
	 * @param feature
	 *            set
	 */
	public void reduceFeatures(Features features) {
		for (Instance inst : this)
			inst.reduceFeatures(features);
	}

	/**
	 * Split this data set into N subsets. The subsets reference the instances
	 * in this set. No copies of instances are made.
	 * 
	 * The set is not changed.
	 * 
	 * @param numNetsTotal
	 * @return
	 */
	public List<DataSet<E>> splitIntoNSubsets(int n) {
		List<DataSet<E>> subsets = new ArrayList<DataSet<E>>(n);
		for (int i = 0; i < n; i++)
			subsets.add(new DataSet<E>());
		int i = 0;
		for (E each : this) {
			i++;
			subsets.get(i % n).addInstance(each);
		}

		return subsets;
	}

	/**
	 * Rename a result feature in all instances.
	 * @param oldFeatureName
	 * @param newFeatureName
	 */
	public void renameResults(String oldFeatureName, String newFeatureName) {
		for (Instance each : this){
			each.putResult(newFeatureName, each.getResult(oldFeatureName));
			each.putResult(oldFeatureName, 0.);
		}
	}

	/**
	 * create a shallow copy of this data set. Instances are not copied.
	 * 
	 * @return
	 */
	public DataSet<E> shallowCopy() {
		DataSet<E> newSet = new DataSet<E>();
		for (E each : this)
			newSet.addInstance(each);
		return newSet;
	}

	/**
	 * get all outcomes as array.
	 * 
	 * @return
	 */
	public double[] outComesAsArray() {
		double[] outcomes = new double[this.size()];
		for (int i = 0; i < size(); i++)
			outcomes[i] = get(i).outcome;
		return outcomes;
	}

	/**
	 * This is equivalent to the SQL "LEFT JOIN USING(id)". Get all the
	 * additional features from the provided data set and merge them with the
	 * instance with the same id in this data set if it exists.
	 * 
	 * @param data
	 */
	public void mergeInstances(DataSet<Instance> data) {
		Map<String, Instance> idMap = new HashMap<String, Instance>();
		for (Instance each : this)
			idMap.put(each.id, each);
		for (Instance each : data)
			if (idMap.containsKey(each.id))
				for (String feature : each.features())
					idMap.get(each.id).put(feature, each.get(feature));
	}

	/**
	 * Get all distinct values of a certain feature. NaN and infinite values
	 * will be ignored.
	 * 
	 * @param feature
	 * @param tolerance
	 *            range within a value is not counted as a distinct value.
	 * @return
	 */
	public List<Double> distinct(String feature, double tolerance) {
		List<Double> di = new ArrayList<Double>();
		for (Instance each : this) {
			double value = each.get(feature);
			if (Double.isNaN(value) || Double.isInfinite(value))
				continue;
			boolean notInList = true;
			for (Double d : di)
				if (Math.abs(value - d) < tolerance)
					notInList = false;
			if (notInList)
				di.add(value);
		}
		return di;
	}

	/**
	 * Save the whole data in a CSV file using only the provided features.
	 * Additionally there will be a column with the outcome and one with the
	 * class.
	 * 
	 * @param features
	 * @param fileName
	 * @throws FileNotFoundException
	 */
	public void saveDataAsCSV(Features features, String fileName,
			String delimiter) throws FileNotFoundException {
		PrintStream ps = new PrintStream(new File(fileName));

		ps.print("class" + delimiter + "outcome");
		for (int i = 0; i < features.size(); i++)
			ps.print(delimiter + features.getFeatureByIndex(i));
		ps.println();

		for (Instance inst : this) {
			ps.print(inst.groundTruth + delimiter + inst.outcome);
			for (int i = 0; i < features.size(); i++)
				ps.print(delimiter + inst.get(features.getFeatureByIndex(i)));
			ps.println();
		}

		ps.close();

	}

	/**
	 * Get the mean value of a certain feature.
	 * @param feature
	 * @return
	 */
	public double mean(String feature) {
		double sum = 0.0;
		for (Instance each : this)
			sum += each.get(feature);
		return sum / this.size();
	}

	/**
	 * calculate the results class rank. All instances have to be classified
	 * including the classification results for all classes.
	 */
	public void calculateRanks(Set<String> classes) {
		for(E e : this){
			List<ClassProbability> ranking = new ArrayList<ClassProbability>();
			for(String className : classes)
				ranking.add(new ClassProbability(className, e.getResult("classProb" + className)));
			Collections.sort(ranking);
			Collections.reverse(ranking);
			double rank = ranking.size();
			for(int i = 0; i < ranking.size(); i++)
				if(e.groundTruth.equals(ranking.get(i).className)){
					rank = i;
					break;
				}
			e.putResult("rank", rank);
		}	
	}

	/**
	 * Equalize the class distribution. All classes will have the same number of
	 * samples in this data set after the application of this function. This is
	 * done by duplicating randomly selected samples from each class until the
	 * number of samples in the respective class is equals the number of samples
	 * in the largest class.
	 */
	@SuppressWarnings("unchecked")
	public void equalizeClassDistribution(int seed) {
		/** get number of samples by class. */
		Map<String, Integer> classSizes = new HashMap<String, Integer>();
		Map<String, List<E>> samplesByClass = new HashMap<String, List<E>>();
		for(E each : this){
			String gt = each.groundTruth;
			if(classSizes.containsKey(gt))
				classSizes.put(gt, classSizes.get(gt) + 1);
			else {
				classSizes.put(gt, 1);
				samplesByClass.put(gt, new ArrayList<E>());
			}
			samplesByClass.get(gt).add(each);
		}
		/** get max number of samples by class. */
		int max = Integer.MIN_VALUE;
		for(Integer i : classSizes.values())
			max = Math.max(max, i);
		
		/** add samples to classes until the have max number of samples. */
		Random rand = new Random(seed);
		for(String className : classSizes.keySet()){
			int n = classSizes.get(className);
			List<E> samples = samplesByClass.get(className);
			for(int i = n; i < max; i++){
				E sample = samples.get(rand.nextInt(n));
				this.add((E) sample.copy());
			}
		}
	}
}

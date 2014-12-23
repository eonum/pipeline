package ch.eonum.pipeline.analysis;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;
import ch.eonum.pipeline.features.FeatureDelta;
import ch.eonum.pipeline.util.Log;

/**
 * <p>Detailed classification. The result of a detailed classification produced by
 * the static function {@link #classify()}.</p>
 * 
 * <p>A rank for the ground truth class is calculated. A rank of 1 is a correct
 * classification. A rank of 2 means the class has the second largest
 * probability/likelihood of all classes and so on.</p>
 * 
 * <p>Optionally for each non zero feature in the provided instance, a likelihood
 * is calculated. A feature likelihood is a measure of how well this feature
 * fits the model of the ground truth class (not the classified class). 0 means
 * no effect. Positive means that this feature increases the
 * possibility/likelihood of the ground truth class whereas negative means a
 * decrease.</p>
 * 
 * <p>The provided classifier must provide probability estimates for all classes in
 * the form "result" + className. Hence all instances have to be classified already.</p>
 * 
 * @author tim
 * 
 */
public class DetailedClassification {

	public static boolean verbose = false;
	/** the classified instance. */
	private SparseInstance instance;
	/** the class ranking. One entry for each class. Ordered by probability. */
	private List<ClassProbability> classRanking;
	/** one entry for each non-zero feature in instance. Ordered by likelihood. */
	private List<FeatureDelta> featuresRanking;
	/** The rank (index in classRanking + 1) of the ground truth class. */
	private int rank;

	/**
	 * Create a detailed classification result.
	 * @param inst the classified instance
	 * @param ranking the class ranking
	 * @param featuresRanking the features by their likelihood
	 */
	public DetailedClassification(SparseInstance inst,
			List<ClassProbability> ranking, List<FeatureDelta> featuresRanking) {
		this.instance = inst;
		this.classRanking = ranking;
		this.featuresRanking = featuresRanking;
		for (int i = 0; i < classRanking.size(); i++)
			if (classRanking.get(i).className.equals(inst.groundTruth))
				this.rank = i + 1;
		if (rank == 0)
			rank = -1;
	}

	public SparseInstance getInstance() {
		return instance;
	}

	public List<ClassProbability> getClassRanking() {
		return classRanking;
	}

	public List<FeatureDelta> getFeaturesRanking() {
		return featuresRanking;
	}

	@Override
	public String toString() {
		return instance.id + ";" + instance.groundTruth + ";" + instance.label
				+ ";" + rank + ";";
	}

	public int getRank() {
		return rank;
	}

	/**
	 * @see {@link DetailedClassification}
	 * @param classifier
	 * @param data
	 * @param classes
	 * @return
	 */
	public static List<DetailedClassification> classify(
			Classifier<SparseInstance> classifier,
			DataSet<SparseInstance> data, Set<String> classes) {
		return DetailedClassification.classify(classifier, data, classes, true);
	}

	/**
	 * @see(DetailedClassification.class)
	 * @param classifier
	 * @param data
	 *            the data to be classified.
	 * @param classes
	 *            set of all possible classes
	 * @param calculateFeatures
	 *            do we have to calculate the deltas for all non-zero features?
	 *            This can be very time consuming. If set to false, all values
	 *            are set to 0.
	 * @return
	 */
	public static List<DetailedClassification> classify(
			Classifier<SparseInstance> classifier,
			DataSet<SparseInstance> data, Set<String> classes,
			boolean calculateFeatures) {
		List<DetailedClassification> results = new ArrayList<DetailedClassification>();
		int h = 0;
		for (SparseInstance inst : data) {
			if (verbose && h++ % 10 == 0)
				Log.puts("Processing instance " + h);
			List<ClassProbability> ranking = new ArrayList<ClassProbability>();
			for (String className : classes)
				ranking.add(new ClassProbability(className, inst
						.getResult("classProb" + className)));
			Collections.sort(ranking);
			Collections.reverse(ranking);

			List<FeatureDelta> featuresRanking = new ArrayList<FeatureDelta>();
			if (calculateFeatures) {
				DataSet<SparseInstance> features = new DataSet<SparseInstance>();
				SparseInstance inst2 = inst.copy();
				inst2.reduceFeatures(classifier.getFeatures());
				for (String feature : inst2.features()) {
					SparseInstance copy = inst2.copy();
					copy.put(feature, 0.);
					copy.id = feature;
					features.add(copy);
				}
				classifier.setTestSet(features);
				features = classifier.test();
				for (Instance f : features) {
					featuresRanking.add(new FeatureDelta(f.id, inst
							.getResult("classProb" + inst.groundTruth)
							- f.getResult("classProb" + inst.groundTruth)));
				}
				Collections.sort(featuresRanking);
			} else {
				for (String feature : inst.features()) {
					featuresRanking.add(new FeatureDelta(feature, 0.));
				}
			}

			results.add(new DetailedClassification(inst, ranking,
					featuresRanking));
		}

		return results;
	}

}

package ch.eonum.pipeline.classification;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ch.eonum.pipeline.clustering.KmeansClustering;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Sequence;
import ch.eonum.pipeline.util.Log;

/**
 * <p>
 * Gaussian Mixture Model with diagonal covariance matrices. The models are
 * trained using Expectation Maximization. This classifiers expects time series
 * (Sequences) in the training and the test data.
 * </p>
 * 
 * <p>
 * All calculations are done in Log-space for efficiency reasons. Issued class
 * likelihoods are transformed to standard likelihoods.
 * </p>
 * 
 * @author tim
 * 
 */
public class GaussianMixtureModel<E extends Sequence> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	private static final double LOG_ZERO = Double.NEGATIVE_INFINITY;
	private static final double MINUS_LOG_THRESHOLD = -39000.14;
	private static final double VARIANCE_TRESHOLD = 0.001;

	static {
		PARAMETERS.put("g", "number of Gaussians (default 1)");
		PARAMETERS
				.put("maxIter",
						"maximum number of iterations in the Expectation Maximization algorithm (default 25)");
	}

	private double[][] logWeights;
	private double[][][] means;
	private double[][][] diagonalCovariance;
	private double[][] constants;
	private Map<String, Integer> classIndexMapping;
	private Map<String, Double> aPrioriClassLogLikelihoods;

	public GaussianMixtureModel(Features features) {
		super();
		this.setFeatures(features);
		this.setSupportedParameters(GaussianMixtureModel.PARAMETERS);
		this.putParameter("g", 1.0);
		this.putParameter("maxIter", 6.0);
	}

	/**
	 * Expectation Maximization.
	 */
	@Override
	public void train() {
		int totalGaussians = (int) this.getDoubleParameter("g");
		Set<String> classes = this.trainingDataSet.collectClasses();
		this.classIndexMapping = new HashMap<String, Integer>();
		int index = 0;
		for (String className : classes)
			this.classIndexMapping.put(className, index++);

		double[] oldTotalG = new double[this.classIndexMapping.size()];
		this.calculateClassAPriori();

		this.initRandom(classes);
		for (int i = 0; i < (int) this.getDoubleParameter("maxIter"); i++) {
			System.out.println("Iteration " + i);
			for (String className : this.classIndexMapping.keySet()) {
				int classIndex = this.classIndexMapping.get(className);
				double[] weights = this.logWeights[classIndex];
				double[] constants = this.constants[classIndex];
				double[][] mns = this.means[classIndex];
				double[][] covDiags = this.diagonalCovariance[classIndex];
				double[] logProbGauss = new double[totalGaussians];
				for (int gauss = 0; gauss < totalGaussians; gauss++)
					logProbGauss[gauss] = GaussianMixtureModel.LOG_ZERO;
				double[][] mu = new double[totalGaussians][features.size()];
				double[][] sigma2 = new double[totalGaussians][features.size()];
				int T = 0;
				double[] w = new double[totalGaussians];
				
				/** Expectation. */
				for (Sequence seq : this.trainingDataSet) {
					if (!seq.groundTruth.equals(className))
						continue;
					double[][] points = seq.getDenseRepresentation(features);
					for (int s = 0; s < seq.getSequenceLength(); s++) {
						double[] point = points[s];
						T++;
						double logPoint = LOG_ZERO;
						double[] logGauss = new double[totalGaussians];
						for (int gauss = 0; gauss < totalGaussians; gauss++) {
							logGauss[gauss] = weights[gauss]
									+ GaussianMixtureModel.N(point, mns[gauss], covDiags[gauss],
											constants[gauss]);
							logPoint = logAdd(logPoint, logGauss[gauss]);
						}
						for (int gauss = 0; gauss < totalGaussians; gauss++) {
							logProbGauss[gauss] = logAdd(logProbGauss[gauss],
									logGauss[gauss] - logPoint);
						}
					}
				}
				
				/** Maximization. */
				for (Sequence seq : this.trainingDataSet) {
					if (!seq.groundTruth.equals(className))
						continue;
					double[][] points = seq.getDenseRepresentation(features);
					for (int s = 0; s < seq.getSequenceLength(); s++) {
						double[] point = points[s];
						double logPoint = LOG_ZERO;
						double[] logGauss = new double[totalGaussians];
						for (int gauss = 0; gauss < totalGaussians; gauss++) {
							logGauss[gauss] = weights[gauss]
									+ GaussianMixtureModel.N(point, mns[gauss], covDiags[gauss],
											constants[gauss]);
							logPoint = logAdd(logPoint, logGauss[gauss]);
						}

						for (int gauss = 0; gauss < totalGaussians; gauss++) {
							double probGauss = Math.exp(logGauss[gauss]
									- logPoint);
							for (int feature = 0; feature < point.length; feature++) {
								mu[gauss][feature] += point[feature]
										* probGauss;
								sigma2[gauss][feature] += Math.pow(
										point[feature] - mns[gauss][feature],
										2.0)
										* probGauss;
							}
						}
					}
				}
				double totalG = LOG_ZERO;

				for (int gauss = 0; gauss < totalGaussians; gauss++) {
					double[] mn = mu[gauss];
					double[] cov = sigma2[gauss];
					double probGauss = Math.exp(logProbGauss[gauss]);
					for (int feature = 0; feature < mn.length; feature++) {
						mn[feature] /= probGauss;
						cov[feature] /= probGauss;
						cov[feature] /= mn.length;
					}
					w[gauss] = logProbGauss[gauss] - T;
					totalG = logAdd(totalG, logProbGauss[gauss]);
				}
				this.means[classIndex] = mu;
				for (int x = 0; x < sigma2.length; x++)
					for (int y = 0; y < sigma2[x].length; y++)
						sigma2[x][y] = sigma2[x][y] < GaussianMixtureModel.VARIANCE_TRESHOLD ? GaussianMixtureModel.VARIANCE_TRESHOLD
								: sigma2[x][y];
				this.diagonalCovariance[classIndex] = sigma2;
				this.logWeights[classIndex] = w;
				double totalWeights = LOG_ZERO;
				for (double weight : this.logWeights[classIndex])
					totalWeights = logAdd(totalWeights, weight);
				for (int iw = 0; iw < this.logWeights[classIndex].length; iw++) {
					this.logWeights[classIndex][iw] -= totalWeights;
				}
				for (int iw = 0; iw < this.logWeights[classIndex].length; iw++)
					System.out.println("Weights for class " + className + ": "
							+ Math.exp(this.logWeights[classIndex][iw]));
				System.out.println("Total Likelihood for class " + className
						+ ": " + totalG + " (" + Math.exp(totalG) + ")"
						+ " Delta: "
						+ (Math.exp(oldTotalG[classIndex]) - Math.exp(totalG)));
				oldTotalG[classIndex] = totalG;
			}
			this.calculateConstants();
		}
	}

	private void calculateClassAPriori() {
		this.aPrioriClassLogLikelihoods = new HashMap<String, Double>();
		for (Instance each : this.trainingDataSet)
			if (!aPrioriClassLogLikelihoods.containsKey(each.groundTruth))
				aPrioriClassLogLikelihoods.put(each.groundTruth, 1.0);
			else
				aPrioriClassLogLikelihoods.put(each.groundTruth,
						aPrioriClassLogLikelihoods.get(each.groundTruth) + 1.0);
		for (String className : aPrioriClassLogLikelihoods.keySet())
			aPrioriClassLogLikelihoods.put(
					className,
					Math.log(aPrioriClassLogLikelihoods.get(className)
							/ this.trainingDataSet.size()));
	}

	private void initRandom(Set<String> classes) {
		int g = (int) this.getDoubleParameter("g");
		this.logWeights = new double[this.classIndexMapping.size()][g];
		this.means = new double[this.classIndexMapping.size()][g][features
				.size()];
		this.diagonalCovariance = new double[this.classIndexMapping.size()][g][features
				.size()];

		for (String cl : classes) {
			DataSet<Instance> set = new DataSet<Instance>();
			for (Sequence seq : this.trainingDataSet)
				if (seq.groundTruth.equals(cl)) {
					set.addAll(seq.createDataSetFromTimePoints());
					if (set.size() > 8000)
						break;
				}
			KmeansClustering<Instance> clusterer = new KmeansClustering<Instance>();
			clusterer.setFeatures(features);
			clusterer.setTrainingSet(set);
			clusterer.putParameter("k", (double) g);
			clusterer.train();
			Map<String, Instance> centers = clusterer.getClusters();
			int classIndex = this.classIndexMapping.get(cl);

			double totalWeights = 0.0;
			for (int i = 0; i < g; i++) {
				this.logWeights[classIndex][i] = Math.random() + 0.1;
				totalWeights += this.logWeights[classIndex][i];
				String clusterName = String.valueOf(i);
				double[] mean = new double[features.size()];
				double[] cov = new double[features.size()];
				Instance center = centers.get(clusterName);
				for (int feature = 0; feature < mean.length; feature++) {
					mean[feature] = center.get(this.features
							.getFeatureByIndex(feature));
					cov[feature] = Math.random() * 10 * GaussianMixtureModel.VARIANCE_TRESHOLD
							+ GaussianMixtureModel.VARIANCE_TRESHOLD;
				}
				this.means[classIndex][i] = mean;
				this.diagonalCovariance[classIndex][i] = cov;
			}
			for (int i = 0; i < g; i++)
				this.logWeights[classIndex][i] = Math
						.log(this.logWeights[classIndex][i] / totalWeights);

			this.calculateConstants();
		}
	}

	private void calculateConstants() {
		int numGauss = (int) this.getDoubleParameter("g");
		this.constants = new double[this.classIndexMapping.size()][numGauss];
		for (String className : this.classIndexMapping.keySet()) {
			int classIndex = this.classIndexMapping.get(className);
			int dim = this.means[0][0].length;
			this.constants[classIndex] = new double[numGauss];
			for (int gauss = 0; gauss < numGauss; gauss++) {
				double determinant = 1.0;
				for (Double each : this.diagonalCovariance[classIndex][gauss])
					determinant *= each;

				double c = Math.log((Math.pow(2 * Math.PI, dim / 2.0) * Math
						.sqrt(determinant)));
				this.constants[classIndex][gauss] = -c;
			}
		}
	}

	/**
	 * Classify the test set. get the classified data set with probability
	 * measures in dimensions for each class: "result" + className.
	 */
	public DataSet<E> test() {
		for (Sequence seq : this.testDataSet) {
			double totalLikelihood = LOG_ZERO;
			String maxClass = "";
			double maxClassLikelihood = LOG_ZERO;
			for (String className : this.classIndexMapping.keySet()) {
				int classIndex = this.classIndexMapping.get(className);
				double[] logWeights = this.logWeights[classIndex];
				double[] constants = this.constants[classIndex];
				double[][] mns = this.means[classIndex];
				double[][] diagCov = this.diagonalCovariance[classIndex];
				double logLikelihood = 0.0;
				double logGaussians[] = new double[logWeights.length];
				for (int gauss = 0; gauss < logWeights.length; gauss++)
					logGaussians[gauss] = LOG_ZERO;
				double[][] points = seq.getDenseRepresentation(features);
				for (int s = 0; s < seq.getSequenceLength(); s++) {
					double[] point = points[s];
					double logGauss = LOG_ZERO;
					for (int gauss = 0; gauss < logWeights.length; gauss++) {
						double n = GaussianMixtureModel.N(point, mns[gauss], diagCov[gauss],
								constants[gauss]);
						if (!Double.isNaN(n)) {
							logGauss = logAdd(logGauss, logWeights[gauss] + n);
							logGaussians[gauss] = logAdd(logGaussians[gauss],
									logWeights[gauss] + n);
						}
					}
					logLikelihood += logGauss;
				}
				/** normalize. */
				for (int gauss = 0; gauss < logWeights.length; gauss++) {
					seq.putResult("gauss" + className + gauss, logGaussians[gauss]);
				}
				logLikelihood = logLikelihood / seq.getSequenceLength();
				logLikelihood += this.aPrioriClassLogLikelihoods.get(className);
				seq.putResult("result" + className, logLikelihood);
				totalLikelihood = logAdd(totalLikelihood, logLikelihood);
				if (maxClassLikelihood < logLikelihood) {
					maxClass = className;
					maxClassLikelihood = logLikelihood;
				}
			}
			seq.label = maxClass;
			/** normalize to sequence length and exponentiate to non-log-likelihoods. */
			for (String className : this.classIndexMapping.keySet()) {
				seq.putResult("result" + className,
						Math.exp(seq.getResult("result" + className)
								- totalLikelihood));
			}
			seq.putResult("result", seq.getResult("result" + maxClass));
		}
		return this.testDataSet;
	}

	/**
	 * Gaussian distribution.
	 * 
	 * @param point
	 * @param center center of the gaussian
	 * @param diagonals
	 *            of the covariance matrix
	 * @return log Gaussian distribution
	 */
	private static double N(double[] x, double[] center, double[] covDiag,
			double constant) {
		assert(x.length == center.length);
		double exponent = 0.0;
		for (int feature = 0; feature < center.length; feature++) {
			exponent += Math.pow(x[feature] - center[feature], 2.0)
					/ covDiag[feature];
		}
		return constant + -0.5 * exponent;
	}

	/**
	 * Add two numbers in log space.
	 * @param summand 1
	 * @param summand 2
	 * @return sum
	 */
	public static double logAdd(double log_a, double log_b) {
		if (log_a < log_b) {
			double tmp = log_a;
			log_a = log_b;
			log_b = tmp;
		}

		double minusdif = log_b - log_a;

		if (minusdif < MINUS_LOG_THRESHOLD)
			return log_a;
		else
			return log_a + Math.log1p(Math.exp(minusdif));
	}

	/**
	 * Subtract two numbers in log space.
	 * @param minuend
	 * @param subtrahend
	 * @return difference
	 */
	public static double logSub(double log_a, double log_b) {
		if (log_a < log_b)
			Log.warn("LogSub: log_a (" + log_a
					+ ") should be greater than log_b (" + log_b + ")");

		double minusdif = log_b - log_a;

		if (log_a == log_b)
			return LOG_ZERO;
		else if (minusdif < MINUS_LOG_THRESHOLD)
			return log_a;
		else
			return log_a + Math.log1p(-Math.exp(minusdif));
	}

}

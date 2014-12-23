package ch.eonum.pipeline.classification.lstm;


import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Sequence;
import ch.eonum.pipeline.util.Log;

/**
 * Thread which evaluates a sub population. Used in {@link GeneticLSTM}. 
 * @author tim
 *
 */
public class SubpopulationEvaluator<E extends Sequence> implements Runnable {

	/** population identifier. */
	private int population;
	/** genes per individual. */
	private double[][][] genes;
	/** current iteration/generation. */
	private int iteration;
	/** fitness of the best performing individual. */
	private double max;
	private GeneticLSTM<E> parent;
	private double[][][] inp;
	private double[][][] tar;
	private double[][][] inp_t;
	private double[][][] tar_t;
	private DataSet<E> dataSet;
	private boolean forgetGates;
	private boolean outputGates;
	private boolean inputGates;

	public SubpopulationEvaluator(int population, int iteration,
			double[][][] genes, GeneticLSTM<E> geneticLSTM, double[][][] inp,
			double[][][] tar, double[][][] inp_t, double[][][] tar_t,
			boolean outputGates, boolean forgetGates, boolean inputGates) {
		this.population = population;
		this.genes = genes;
		this.iteration = iteration;
		this.parent = geneticLSTM;
		this.inp = inp;
		this.tar = tar;
		this.inp_t = inp_t;
		this.tar_t = tar_t;
		this.dataSet = parent.getTestDataSet().deepCopy();
		this.outputGates = outputGates;
		this.forgetGates = forgetGates;
		this.inputGates = inputGates;
	}

	@Override
	public void run() {
		max = Double.NEGATIVE_INFINITY;
		int maxI = -1;
		for(int i = 0; i < genes.length; i++){
			double eval = parent.evaluate(genes[i], iteration,
					population, i, dataSet, inp_t, tar_t);
			if(eval > max){
				max = eval;
				maxI = i;
			}
		}
		Log.puts("Best gene from population " + population + " without retraining: " + max);
		parent.println("Best gene from population " + population + " without retraining: " + max);
		
		/** retrain the net. */
		LSTMCore<E> net = new LSTMCore<E>("Population" + population,
				parent.getBaseDir() + "genetic" + population + "/", parent, 0, outputGates, forgetGates, inputGates, false);
		net.setTrainingData(inp, tar);
		net.setTestData(inp_t, tar_t);
		net.setWeightMatrix(genes[maxI]);
		net.train();
		
		max = parent.evaluate(net.getWeightMatrix(), iteration,
				population, -1, dataSet, inp_t, tar_t);
		
		Log.puts("Best gene from population " + population + " after retraining: " + max);
		parent.println("Best gene from population " + population + " after retraining: " + max);
				
		this.genes[0] = net.getWeightMatrix();	
	}

	public double getMaxIndividualPerformance() {
		return this.max;
	}

}

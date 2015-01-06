package ch.eonum.pipeline.core;

/**
 * Sequence data set. Extends DataSet by sequence specific transformations.
 * 
 * @author tim
 *
 * @param <E>
 */
public class SequenceDataSet<E extends Sequence> extends DataSet<E> {

	private static final long serialVersionUID = -6190330850640598339L;
	
	/**
	 * Create from standard data set. Many readers read standard data sets only
	 * and hence have to be transformed into a SequenceDataSet
	 * 
	 * @param data
	 */
	public SequenceDataSet(DataSet<E> data) {
		this.addAll(data);
	}

	public SequenceDataSet() {}

	/**
	 * Level/Flat the time series data. By leveling, a sequence can be used with
	 * standard classifiers which cannot process time series. Level Average, Max
	 * and Min. 
	 */
	public void levelSequences() {
		for (Sequence each : this) {
			each.levelAverage();
			each.levelMax();
			each.levelMin();
		}
	}

	/**
	 * Level/Flat the time series data. By leveling, a sequence can be used with
	 * standard classifiers which cannot process time series. Level Average,
	 * Max, Min and Standard deviation
	 */
	public void levelSequencesAll() {
		for (Sequence each : this) {
			each.levelAverage();
			each.levelMax();
			each.levelMin();
			each.levelStd();
		}
	}

	/**
	 * Level/Flat the time series data. By leveling, a sequence can be used with
	 * standard classifiers which cannot process time series. Level Average,
	 * Max, Min, standard deviation and center of mass
	 */
	public void levelSequencesAllwithCOM(Features features) {
		for (Sequence each : this) {
			each.levelAverage();
			each.levelMax();
			each.levelMin();
			each.levelStd();
			each.levelCenterOfMass(features);
		}
	}

	/**
	 * Level/Flat the time series data. By leveling, a sequence can be used with
	 * standard classifiers which cannot process time series. Level a time
	 * window: precondition: every sequence must have the same length. For each
	 * element in the sequence a new set of features is created.
	 * 
	 * @see Sequence#levelTimeWindow()
	 */
	public void levelSequencesTimeWindow() {
		for (Sequence each : this) {
			each.levelTimeWindow();
		}
	}

	/**
	 * Inverse operation to levelSequences. Push all data in the flat instance
	 * to each time point in a sequence.
	 * 
	 * @see Sequence#pushUp()
	 */
	public void pushUp() {
		for (Sequence each : this) {
			each.pushUp();
		}
	}

	/**
	 * Delete the time series data.
	 */
	public void deleteSequence() {
		for (Sequence each : this) {
			each.deleteSequence();
		}
	}
	
	/**
	 * Return a data set with all sequences unrolled as instances. No copies of
	 * data is made. Hence operations on the resulting data set affect this
	 * data set. This is convenient to apply feature transformations like
	 * normalizations, which have been designed for instance data sets, on
	 * sequence data.
	 * 
	 * @see Sequence#getDataSetFromTimePoints()
	 * 
	 * @return
	 */
	public DataSet<SparseInstance> createDataSetFromSequences() {
		DataSet<SparseInstance> ds = new DataSet<SparseInstance>();
		for (Sequence each : this)
			ds.addAll(each.getDataSetFromTimePoints());
		return ds;
	}
	
	/**
	 * invert the temporal order of all sequences in this data set.
	 * 
	 * @see Sequence#invertTemporalOrder()
	 */
	public void invertSequences() {
		for (Sequence each : this)
			each.invertTemporalOrder();
	}

	/**
	 * push the master (instance) data on top of the sequence data.
	 * Precondition: all instances in this data set are sequences.
	 * 
	 * @see Sequence#pushMasterOnTop()
	 */
	public void pushMasterOnTop() {
		for (Sequence each : this)
			each.pushMasterOnTop();
	}

	/**
	 * put the master (instance) data at the bottom of the sequence data.
	 * Precondition: all instances in this data set are sequences.
	 * 
	 * @see Sequence#putMasterAtBottom()
	 */
	public void putMasterAtBottom() {
		for (Sequence each : this)
			each.putMasterAtBottom();
	}

	/**
	 * Add a copy of the sequence at the end of the sequence. Some extra context
	 * in both directions can be obtained by this.
	 * 
	 * @see Sequence#duplicateSequence()
	 */
	public void duplicateSequences() {
		for (Sequence each : this)
			each.duplicateSequence();
	}

	/**
	 * create a target sequence for each sequence by copying the sequence with a
	 * given time lag. Hence we can can train to predict the sequence. The last
	 * timeLag points will be NaN, because we do not know the targets.
	 * 
	 * @see Sequence#createTargetForPrediction(int, Features)
	 * 
	 * @param timeLag
	 *            in number of points in time.
	 * @param features
	 *            features which should be predicted. Usually the same set as
	 *            the input feature set.
	 */
	public void setTimeLag(int timeLag, Features features) {
		for (Sequence each : this)
			each.createTargetForPrediction(timeLag, features);
	}

}

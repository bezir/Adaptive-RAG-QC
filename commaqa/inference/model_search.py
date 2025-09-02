import copy
import heapq
import json
import logging

from commaqa.inference.data_instances import BasicDataInstance


class ParticipantModel(object):
    """Base model in this case for coordinating different models. Provides a general
    class to structure all contributing models (in this case, by defining a single
    function `query`, which is the single method that is called for each model).

    """

    def query(self, state, debug=False):
        """The main function that interfaces with the overall search and
        model controller, and manipulates the incoming data.

        :param state: the state of controller and model flow.
        :type state: launchpadqa.question_search.model_search.SearchState
        :rtype: list
        """
        raise NotImplementedError("Must implement to work inside of controller!")

    def return_model_calls(self):
        """
        :return: a dict of <model_name, number of calls> made by this participant
        """
        raise NotImplementedError("Must implement to work inside of controller!")


class ModelController(object):
    """This class is a `ModelController` that takes multiple (arbitrary)
    models and a control specification of how to interface the different
    models (which can be thought of as a kind of state graph). For example

    """

    def __init__(self, model_list, data_class=BasicDataInstance):
        """Create an instance of a ComplexModel

        :param model_list: a list of models with identifiers and
          control flow.
        :type model_list: dict
        """
        if "start_state" not in model_list:
            raise ValueError("Must specify start state")
        if "end_state" not in model_list:
            raise ValueError("Must specify end state")
        self.model_list = model_list
        self.data_class = data_class
        self.logger.setLevel(logging.ERROR)  # Suppress debug/info messages

    def execute(self, state, debug=False):
        """Executes a command and query

        :param state: a given state in search
        :type state: SearchState (defined here)
        :returns: a list of output
        :rtype: list
        """
        if state.next not in self.model_list:
            # CRITICAL FIX: Handle None state.next properly to prevent TypeError
            next_state_str = str(state.next) if state.next is not None else "None"
            self.logger.error(f"❌ ERROR: Cannot handle next state: '{next_state_str}'")
            self.logger.error(f"🔧 DEBUG: Available models: {list(self.model_list.keys())}")
            self.logger.error(f"🔧 DEBUG: Current state info - next: {state.next}, score: {getattr(state, 'score', 'N/A')}")
            
            # Add detailed state analysis
            if hasattr(state, 'data') and state.data:
                self.logger.error(f"🔧 DEBUG: State data keys: {list(state.data.__dict__.keys()) if hasattr(state.data, '__dict__') else 'No __dict__'}")
                if hasattr(state.data, 'generated_sentences'):
                    sentences = getattr(state.data, 'generated_sentences', [])
                    self.logger.error(f"🔧 DEBUG: Generated sentences count: {len(sentences) if sentences else 0}")
                    if sentences:
                        self.logger.error(f"🔧 DEBUG: Last sentence: '{sentences[-1] if sentences else 'None'}'")
            
            return []
        
        # Add comprehensive state transition debug prints
        self.logger.debug(f"🔧 DEBUG: ====== MODEL EXECUTION: {state.next} ======")
        self.logger.debug(f"🔧 DEBUG: State transition: executing model '{state.next}'")
        if hasattr(state, 'data') and state.data:
            if hasattr(state.data, 'generated_sentences'):
                sentences = getattr(state.data, 'generated_sentences', [])
                self.logger.debug(f"🔧 DEBUG: Current generated sentences count: {len(sentences) if sentences else 0}")
        
        try:
            model_func = self.model_list[state.next]
            self.logger.debug(f"🔧 DEBUG: About to execute model function for '{state.next}'")

            model_output = model_func(state, debug=debug)

            # Add detailed output analysis
            if isinstance(model_output, list):
                self.logger.debug(f"🔧 DEBUG: Model '{state.next}' returned {len(model_output)} output states")
                for i, output_state in enumerate(model_output):
                    if hasattr(output_state, 'next'):
                        next_model = output_state.next if output_state.next is not None else "None"
                        self.logger.debug(f"🔧 DEBUG: Output state {i+1}: next='{next_model}'")
                    if hasattr(output_state, 'data') and output_state.data:
                        if hasattr(output_state.data, 'generated_sentences'):
                            sentences = getattr(output_state.data, 'generated_sentences', [])
                            sentence_count = len(sentences) if sentences else 0
                            self.logger.debug(f"🔧 DEBUG: Output state {i+1}: {sentence_count} sentences")
                            if sentences and sentence_count > 0:
                                last_sentence = sentences[-1] if sentences else "None"
                                self.logger.debug(f"🔧 DEBUG: Output state {i+1}: last sentence: '{last_sentence[:100]}...'")
            else:
                self.logger.debug(f"🔧 DEBUG: Model '{state.next}' returned non-list output: {type(model_output)}")
                # Handle non-list outputs by wrapping in list
                if model_output is not None:
                    model_output = [model_output]
                else:
                    self.logger.debug(f"🔧 DEBUG: Model '{state.next}' returned None - converting to empty list")
                    model_output = []
            
            self.logger.debug(f"🔧 DEBUG: ====== MODEL EXECUTION COMPLETED: {state.next} ======")
            return model_output
        except Exception as e:
            self.logger.debug(f"Model execution failed: {e}")  # Suppress stack trace spam
            raise ValueError("Error caught during model execution:  %s" % e)

    def init_data(self, data_instance):
        """Create an initialized version of the data object
        that will get through around.

        :param data_instance: any arbitrary piece of data.
        :rtype: self.data_class
        """
        return self.data_class(data_instance)

    @property
    def start_state(self):
        return self.model_list["start_state"]

    @property
    def end_state(self):
        return self.model_list["end_state"]

    @property
    def logger(self):
        """Returns a logger instance"""
        level = ".".join([__name__, type(self).__name__])
        return logging.getLogger(level)


## utility class for controlling and recording search state


class SearchState(object):
    """Tracks and records the state of a given search."""

    def __init__(self, json_data, command, score=0.0):
        """Keep track of different stages in the state

        :param json_data: some basic, json represntation of data
        """
        self._data = json_data
        self._score = score
        self._next = command

    def copy(self):
        """Does a deep copy of the state

        :returns: new search state
        """
        new_data = copy.deepcopy(self._data)
        new_score = copy.deepcopy(self._score)
        new_next = copy.deepcopy(self._next)

        return SearchState(new_data, new_next, new_score)

    ## important to implement to work
    ## with the heap datastructures
    def __lt__(self, other):
        if self.score < other.score:
            return True
        return False

    def __eq__(self, other):
        if self.score == other.score:
            return True
        return False

    @property
    def data(self):
        return self._data

    @property
    def score(self):
        return self._score

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, value):
        self._next = value

    @data.setter
    def data(self, value):
        self._data = value


class QuestionSearchBase(object):
    def __init__(self, model_controller):
        """Create a `QuestionDecomposer instance`

        :param model_ensemble: a collection of models with control instructions
        """
        self.controller = model_controller

    def find_answer_decomp(self, json_input, debug=False):
        """Main question decomposition function

        :param json_input: the input to all of the models.
        """
        raise NotImplementedError

    def return_qid_prediction(
        self,
        example,
        override_answer_by=None,
        debug=False,
        silent=False,
    ):
        final_state, other_states = self.find_answer_decomp(example, debug=debug)
        if final_state is None:
            if not silent:
                print(example["question"] + " FAILED!")
            chain = "\n" + example["qid"] + "\n" + example["question"]
            if not silent:
                print("\n")
            return (example["qid"], "", chain)
        else:
            data = final_state._data
            chain = "\n" + example["qid"] + "\n" + example["question"]
            chain += "\n" + data.get_printable_reasoning_chain()
            chain += "\nS: " + str(final_state._score)
            if not silent:
                print(chain)
            if override_answer_by is not None:
                if override_answer_by not in data:
                    print(f"WARNING: The key {override_answer_by} is not present in the data dict.")
                final_answer = data.get(override_answer_by, "")
                if not isinstance(final_answer, str):
                    final_answer = json.dumps(final_answer)
            else:
                final_answer = data.get_last_answer()
            try:
                json_answer = json.loads(final_answer)
                # use this only if list (ignore numbers, etc)
                if isinstance(json_answer, list) or isinstance(json_answer, str):
                    final_answer = json_answer
            except ValueError:
                # Not a valid json ignore
                pass
            if not silent:
                print("\n")
            return (example["qid"], final_answer, chain)


class BestFirstDecomposer(QuestionSearchBase):
    def find_answer_decomp(self, json_input, debug=False):
        """Run the question decomposer. The main function here is to use
        the controller to pass around inputs to the different models, then
        keep a track of the search state and terminate when the shortest path
        has been found.

        :param json_input: some input to the model
        """
        ## start state of controller : e.g., generate
        start_command = self.controller.start_state
        start_data = self.controller.init_data(json_input)

        ## min-heap
        heap = []
        init_input = json_input["question"] if json_input["question"] else "UNKNOWN"
        if debug:
            print("[START QUERY] : %s" % init_input)

        init_state = SearchState(
            start_data,  ## initial input
            start_command,  ## starting point
            score=0.0,  ## starting score
        )

        ## push it to heap
        heapq.heappush(heap, init_state)

        ## start the main search
        iteration_count = 0
        max_iterations = 15  # INCREASED: Allow more retrieve→generate cycles for complex multi-hop questions
        while True:
            iteration_count += 1
            if iteration_count > max_iterations:
                if debug:
                    print(f"[MAX_ITERATIONS_REACHED]: {init_input} after {max_iterations} iterations")
                # Return best state found so far if any
                if len(heap) > 0:
                    return heapq.heappop(heap), heap
                return None, []
            if len(heap) == 0:
                if debug:
                    print("[FAILED]: %s" % init_input)
                return None, []

            ## pop from heap
            current_state = heapq.heappop(heap)

            if debug:
                print("[MIN_STATE] command=%s" % (current_state.next))
            # if current_state.next is None:
            # print(current_state.data.get_printable_reasoning_chain())
            #     current_state.next = current_state.data.get_last_generator()
            ## end state
            if current_state.next == self.controller.end_state:
                if current_state.data.has_tasks():
                    new_task = current_state.data.pop_task()
                    # print("popped task!")
                    # print(new_task)
                    new_state = current_state.copy()
                    if new_task.task_question:
                        new_state.data.add_qgen(new_task.task_question)
                    new_state.next = new_task.task_participant
                    heapq.heappush(heap, new_state)
                    continue
                else:
                    if debug:
                        print("[TERMINATED]")
                    return current_state, heap

            ## generate output and new stated
            for new_state in self.controller.execute(current_state, debug=debug):

                ## push onto heap
                heapq.heappush(heap, new_state)

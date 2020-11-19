import torch
import json

model_path = "resources/bertQA_model"
tokenizer_path = "resources/bertQA_tokenizer"

model = torch.load(model_path)
tokenizer = torch.load(tokenizer_path)

text = "Customers require the authorization object LICAUD_CLO on their S-user id in order to access the Licenses information. SAP colleagues can access this information without further authorization. The license utilization data is updated every day at around 9:00 (CET) from the License Utilization Information Application (LUI) and loads the data for the last 30 days including the actual day, in this way if certain data was missed/incorrect on a certain day over the last 30 days, the daily update will correct this data. Note: the LUI also has a refresh schedule that you should confirm if needing to have that granularity. Some metrics are only displayed internally. The reason for this is the external version of the cockpit should align with the LUI tool and the internal tool should align with the Cloud Reporting Tool. This includes metrics that are unlimited, these are not shown in LUI and therefor will only be seen internally. The chart or table in this subsection shows the license utilization for the different components of your cloud solutions, according to the license metrics applicable to each. All product counts include only valid, non-deleted forms. In order to access Licenses area, the authorization object LICAUD_CLO is required for your S-user id. Reach out to your super administrator if required. You can switch to table-only view to see more details. By clicking one of the metrics on the chart or in the table, you can see the history of the purchased and used licenses in the License Utilization Trend subsection on the right."

def bert(question):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, text)

    # Report how long the input sequence is.
    #print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return json.loads("' + answer + '")
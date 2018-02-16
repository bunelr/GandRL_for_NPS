def add_common_arg(parser):
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use the GPU to run the model")
    parser.add_argument("--log_frequency", type=int,
                        default=100,
                        help="How many minibatch to do before logging"
                        "Default: %(default)s.")

def s2intL(str_args):
    return [int(val) for val in str_args.split(',')]

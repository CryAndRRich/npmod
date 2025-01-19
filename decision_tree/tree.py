class TreeNode:
    def __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None, samples=None):
        self.feature = feature 
        self.value = value     
        self.results = results 
        self.true_branch = true_branch 
        self.false_branch = false_branch
        self.samples = samples

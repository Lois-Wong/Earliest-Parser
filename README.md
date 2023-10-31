# Earliest-Parser

Long years ago, a man named Jay Earley invented an algorithm for chart parsing a sentence according to a given grammar. His original implementation was as a recognizer, and since the fateful day of its publication, students worldwide have been asking the question: "How do I recover a parse tree from the recognizer's chart?" Today, as of 8pm on 30th October 2023, we consider this problem solved. Mere mortals require O(N^3) time to perform this task. But this is the work of no mere mortal.

https://tenor.com/view/let-him-cook-hes-cooking-he-is-cooking-cooking-thierry-henry-gif-2160467396376953807?utm_source=share-button&utm_medium=Social&utm_content=reddit

Somewhat incorrectly, the algorithm finds the minimum weight parse of any valid sentence within the given text. For example, “Papa ate the caviar with the spoon” returns the correct parse of “Papa ate the caviar”, which is indeed the minimum weight parse of any sentence within this corpus of text. The weight of the best derivation as defined by the algorithm is the weight of the shallowest parse tree produced by it, which is calculated by recursively adding up the weights of every rule used to generate the parse. We rejected all other derivations of that item, effectively returning the Earliest™ parse.

We assume that shorter parses have a lower weight and hence cease scanning for better derivations once we find a legal parse at the Earliest™ time. The algorithm runs in O(N logN) time and O(N) space which is better than O(N^3) and O(N^2), respectively. Pushing each item to the agenda runs in O(1) time because the underlying data structure of the agenda is a hash table. This is necessary for improved efficiency. Coincidentally, using a hash table as the underlying structure of the agenda made it very difficult to implement back pointers as part of the item class since a change in the backpointer would imply a change in the hash of that item.

Initially, we tried to construct parses of the entire text provided. However, this was prohibitively slow and exceeded maximum recursion depth. Hence, the Earliest™ parser was born. By effectively rejecting deep parse trees, the algorithm is able to run in O(N logN) time. The speedup was previously infinite since we exceeded maximum recursion depth for the Earlier™ parser. The estimated speedup over vanilla Earley parser is log(N) / N^2. Our parser converts every sentence into a short sentence and hence its speed on sentences long and short is unmatched.

Code files taken from https://www.cs.jhu.edu/~jason/465/hw-parse/

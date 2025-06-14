class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        """Add a node to the end of the list."""
        new_node = Node(data)
        
        if self.head is None:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def print_list(self):
        """Print all nodes in the list."""
        if self.head is None:
            print("List is empty")
            return
        
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")
    
    def delete_nth_node(self, n):
        """
        Delete the nth node from the list (1-based indexing).
        Raises IndexError if n is out of range or list is empty.
        """
        if self.head is None:
            raise IndexError("Cannot delete from an empty list")
        
        if n < 1:
            raise IndexError("Index must be positive")
        
        if n == 1:
            self.head = self.head.next
            return
        
        current = self.head
        count = 1
        
        while current and count < n - 1:
            current = current.next
            count += 1
        
        if current is None or current.next is None:
            raise IndexError("Index out of range")
        
        current.next = current.next.next

if __name__ == "__main__":
    ll = LinkedList()
    
    ll.append(1)
    ll.append(2)
    ll.append(3)
    ll.append(4)
    ll.append(5)
    
    print("Original list:")
    ll.print_list()
    
    try:
        ll.delete_nth_node(3)
        print("\nAfter deleting 3rd node:")
        ll.print_list()
    except IndexError as e:
        print(f"Error: {e}")
    
    try:
        ll.delete_nth_node(1)
        print("\nAfter deleting 1st node:")
        ll.print_list()
    except IndexError as e:
        print(f"Error: {e}")
    
    try:
        ll.delete_nth_node(10)
    except IndexError as e:
        print(f"\nError when trying to delete 10th node: {e}")
    
    empty_ll = LinkedList()
    try:
        empty_ll.delete_nth_node(1)
    except IndexError as e:
        print(f"Error when trying to delete from empty list: {e}") 
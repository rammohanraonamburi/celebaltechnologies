def print_lower_triangle(n):
    """
    Print a lower triangular pattern of asterisks
    Example for n=4:
    *
    **
    ***
    ****
    """
    for i in range(1, n + 1):
        print('*' * i)

def print_upper_triangle(n):
    """
    Print an upper triangular pattern of asterisks
    Example for n=4:
    ****
    ***
    **
    *
    """
    for i in range(n, 0, -1):
        print('*' * i)

def print_pyramid(n):
    """
    Print a pyramid pattern of asterisks
    Example for n=4:
       *
      ***
     *****
    *******
    """
    for i in range(1, n + 1):
        # Print spaces
        print(' ' * (n - i), end='')
        # Print asterisks
        print('*' * (2 * i - 1))

def main():
    size = 4
    print("Lower Triangle Pattern:")
    print_lower_triangle(size)
    print("\nUpper Triangle Pattern:")
    print_upper_triangle(size)
    print("\nPyramid Pattern:")
    print_pyramid(size)

if __name__ == "__main__":
    main() 
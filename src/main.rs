use core::panic;
use std::thread::current;

use std::collections;

// https://rust-unofficial.github.io/patterns/patterns/behavioural/visitor.html

// TODO: could extend number to integer, float, etc.
//      maybe from that figure out traits to more easily perform operations
// TODO: remove pub, i think only needed because was in some namespace before
pub enum Node {
    None,
    Number(f32),
    BinaryAdd([Box<Node>; 2]),
    BinarySub([Box<Node>; 2]),
    BinaryMul([Box<Node>; 2]),
    BinaryDiv([Box<Node>; 2]),
    Variable(String),
    Compound(Vec<Node>)
}

// TODO: store memory in here
struct InterpreterContext {

}

// TODO: Store token stream, token index, memory
struct ParserContext {

}

// TODO: Store code string, string index
struct LexerContext {

}

// TODO: proper "advance" and "peek" functions for lexer/parser/interpreter
// TODO: multi-file project probably soon (or at least namespaces for lexer/parser/interpreter)

// Switch and call correct visitor method
fn visit(n: &Node) -> Node {
    match n {
        Node::BinaryAdd(children) => return visit_add(&children[0], &children[1]),
        Node::BinarySub(children) => return visit_sub(&children[0], &children[1]),
        Node::BinaryMul(children) => return visit_mul(&children[0], &children[1]),
        Node::BinaryDiv(children) => return visit_div(&children[0], &children[1]),
        Node::Compound(children) => return visit_compound(children),
        Node::Number(value) => return Node::Number(*value),
        Node::Variable(string) => return visit_variable(string),
        Node::None => return Node::None,
        _ => panic!("Couldn't extract value of node")
    }
}

fn visit_variable(string : &String) -> Node {
    // TODO: need to pass in memory, and then lookup variable in memory
    
    return Node::Number(5.0);
}

fn visit_compound(children : &Vec<Node>) -> Node {
    let mut last_node : Node = Node::None;
    
    for child in children {
        last_node = visit(child);
    }

    return last_node;
}

fn visit_binary_number(left: &Node, right: &Node) -> [f32; 2] {
    let visited_left : Node = visit(left);
    let visited_right : Node = visit(right);

    // Attempt to convert both to number types
    if let Node::Number(left_value) = visited_left {
        if let Node::Number(right_value) = visited_right {
            return [left_value, right_value];
        }
    }

    panic!("One side of binary add was invalid");
}

fn visit_add(left: &Node, right: &Node) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right);

    return Node::Number(values[0] + values[1]);
}

fn visit_sub(left: &Node, right: &Node) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right);

    return Node::Number(values[0] - values[1]);
}

fn visit_div(left: &Node, right: &Node) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right);

    return Node::Number(values[0] / values[1]);
}

fn visit_mul(left: &Node, right: &Node) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right);

    return Node::Number(values[0] * values[1]);
}

enum Token {
    None,
    Number(f32),
    Operator(u8),
    Assign,
    Identifier(String),
    StatementEnd
}

fn read_number(string : &String, pos : &mut usize) -> Token {
    let mut current_value : f32 = 0.0;
    let mut digits_after_decimal : i32 = 0;
    
    while *pos < string.len() {
        let current_char : u8 = string.as_bytes()[*pos];

        match current_char {
            b'0'..=b'9' => {
                let digit_value = current_char - b'0';
                
                if digits_after_decimal == 0 {
                    current_value *= 10.0;
                    current_value += digit_value as f32;
                } else {
                    current_value += digit_value as f32 * (10 as f32).powf(-digits_after_decimal as f32) as f32;
                    digits_after_decimal += 1;
                }
            }
            b'.' => {
                if digits_after_decimal > 0 {
                    panic!("Multiple decimal points inside number");
                }

                digits_after_decimal = 1;
            }
            _ => return Token::Number(current_value)
        }

        *pos += 1;
    }

    if current_value != 0.0 {
        return Token::Number(current_value);
    }
    
    panic!("Didn't read integer, or reached EOF");
}

fn read_identifier(string : &String, pos : &mut usize) -> Token {
    let start: usize = *pos;
    
    while *pos < string.len() {
        let current_char = string.as_bytes()[*pos];

        if current_char.is_ascii_alphabetic() {
            *pos += 1;
        } else {
            break;
        }
    }

    let identifier: String = String::from(&string[start..*pos]);
    
    *pos -= 1;

    return Token::Identifier(identifier);
}

fn lex(string : &String) -> Vec<Token> {
    let mut tokens : Vec<Token> = Vec::new();

    let mut current_pos : usize = 0;

    while current_pos < string.len() {
        let current_char : u8 = string.as_bytes()[current_pos];

        // Read identifier
        if current_char.is_ascii_alphabetic() {
            tokens.push(read_identifier(string, &mut current_pos));
        } else {
            match current_char {
                b'0'..=b'9' | b'.' => {
                    tokens.push(read_number(string, &mut current_pos));
                    current_pos -= 1;
                }
                b'+' | b'-' | b'*' | b'/' => {
                    tokens.push(Token::Operator(current_char));
                }
                b'=' => {
                    tokens.push(Token::Assign);
                }
                b';' => {
                    tokens.push(Token::StatementEnd);
                }
                _ => {}
            }
        }

        current_pos += 1;
    }

    return tokens;
}


// TODO: parser shouldn't have to access memory other than to assign value
fn factor(token_index : &mut usize, token_stream : &Vec<Token>, memory : &mut collections::BTreeMap<String, Node>) -> Node {
    if let Token::Number(value) = token_stream[*token_index] {
        *token_index += 1;

        return Node::Number(value);
    }
    else if let Token::Identifier(name) = &token_stream[*token_index] {
        *token_index += 1;

        return Node::Variable(String::from(name));
    }

    panic!("Expected number or identifier");
}

fn term(token_index : &mut usize, token_stream : &Vec<Token>, memory : &mut collections::BTreeMap<String, Node>) -> Node {
    let mut lhs: Node = factor(token_index, token_stream, memory);

    if *token_index >= token_stream.len() {
        return lhs;
    }

    let mut current_token : &Token = &token_stream[*token_index];
    
    while *token_index < token_stream.len() {
        current_token = &token_stream[*token_index];

        if let Token::Operator(operator_type) = current_token {
            match operator_type {
                b'*' => {
                    *token_index += 1;
                    lhs = Node::BinaryMul([Box::new(lhs), Box::new(factor(token_index, token_stream, memory))]);
                },
                b'/' => {
                    *token_index += 1;
                    lhs = Node::BinaryDiv([Box::new(lhs), Box::new(factor(token_index, token_stream, memory))]);
                },
                _ => return lhs
            }
        } else {
            return lhs;
        }
    }

    return lhs;
}

fn expression(token_index : &mut usize, token_stream: &Vec<Token>, memory : &mut collections::BTreeMap<String, Node>) -> Node {
    let mut lhs: Node = term(token_index, token_stream, memory);

    if *token_index >= token_stream.len() {
        return lhs;
    }

    let mut current_token : &Token = &token_stream[*token_index];
    
    while *token_index < token_stream.len() {
        current_token = &token_stream[*token_index];

        if let Token::Operator(operator_type) = current_token {
            match operator_type {
                b'+' => {
                    *token_index += 1;
                    lhs = Node::BinaryAdd([Box::new(lhs), Box::new(term(token_index, token_stream, memory))]);
                },
                b'-' => {
                    *token_index += 1;
                    lhs = Node::BinarySub([Box::new(lhs), Box::new(term(token_index, token_stream, memory))]);
                },
                _ => return lhs
            }
        } else {
            return lhs;
        }
    }

    return lhs;
}

const NULL_TOKEN : Token = Token::None;

fn peek<'a>(token_index : &mut usize, token_stream : &'a Vec<Token>) -> &'a Token {
    if *token_index + 1 < token_stream.len() {
        let peek_token : &Token = &token_stream[*token_index + 1];

        return peek_token;
    }
    
    return &NULL_TOKEN;
}

fn statement(token_index : &mut usize, token_stream: &Vec<Token>, memory : &mut collections::BTreeMap<String, Node>) -> Node {
    while *token_index < token_stream.len() {
        let current_token : &Token = &token_stream[*token_index];

        // TODO: badly formed right now
        match current_token {
            Token::Identifier(name) => {
                let next_token : &Token = peek(token_index, token_stream);
                
                match next_token {
                    Token::Assign => {
                        *token_index += 2;

                        let rhs: Node = expression(token_index, token_stream, memory);

                        memory.insert(name.to_string(), rhs);

                        return Node::Variable(String::from(name));
                    },
                    _ => {
                        return expression(token_index, token_stream, memory);
                    }
                }
            } 
            Token::Number(_) => return expression(token_index, token_stream, memory),
            _ => return Node::None
        }

        // *token_index += 1;
    }

    return Node::None;
}

// Build an AST from a token stream
// Recursive descent parser
fn parse(token_index : &mut usize, token_stream: &Vec<Token>, memory : &mut collections::BTreeMap<String, Node>) -> Node {
    let mut statements : Vec<Node> = Vec::new();
    
    while *token_index < token_stream.len() {
        // Read a statement
        statements.push(statement(token_index, token_stream, memory));

        if *token_index < token_stream.len() {
            let current_token : &Token = &token_stream[*token_index];
        
            // Expect an end token
            if let Token::StatementEnd = current_token {
                *token_index += 1;
            }
            else { break; }
        }
    }
    
    return Node::Compound(statements);
}

fn interpret(tree: &Node) -> Node {
    return visit(tree);
}

fn main() {
    let mut memory : collections::BTreeMap<String, Node> = collections::BTreeMap::new();
    let input : String = String::from("x=5+3;x");
    let tokens: Vec<Token> = lex(&input);
    let mut token_index : usize = 0;
    let tree: Node = parse(&mut token_index, &tokens, &mut memory);
    let result : Node = interpret(&tree);

    if let Node::Number(value) = result {
        println!("Res: {value}");
    } else {
        panic!("Interpreter didn't return number");
    }
}
